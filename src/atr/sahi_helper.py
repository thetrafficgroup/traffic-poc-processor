"""SAHI-style sliced inference for the ATR pipeline.

Opt-in via env var `ATR_USE_SAHI=1`. When enabled, `sliced_predict` is used in
place of `model.predict` at the two detection call sites in `atr_processor.py`.

The winning S20 configuration (process-tweaks.md, Section 10):
  - Tile a lane-aware band (derived from lane polygons) with 320x240 tiles,
    25% overlap, imgsz=1280 per tile, no TTA on tiles.
  - Run the full frame at imgsz=1408 with TTA (augment=True).
  - Merge all detections via per-class NMS at iou=0.3.

Measured deltas vs iter 19 baseline on the two validation clips:
  multi_2 abs_err: 17 -> 8 (-53%)
  multi_3 abs_err: 30 -> 12 (-60%)

Compute cost: ~5x baseline GPU time per frame.

The helper returns objects that mimic the ultralytics `Results` shape consumed
downstream (iterable `.boxes` with `.xyxy`, `.conf`, `.cls`, indexing by
tensor for the per-class filter, etc).
"""

import torch
import torchvision


class _FakeBox:
    def __init__(self, xyxy, conf, cls, device):
        self.xyxy = torch.tensor([xyxy], device=device, dtype=torch.float32)
        self.conf = torch.tensor([conf], device=device, dtype=torch.float32)
        self.cls = torch.tensor([cls], device=device, dtype=torch.float32)

    @property
    def xywh(self):
        x1, y1, x2, y2 = self.xyxy[0].tolist()
        return torch.tensor(
            [[(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]],
            device=self.xyxy.device,
        )


class _FakeBoxes:
    def __init__(self, boxes_list, device):
        self._boxes = list(boxes_list)
        self._device = device
        self._rebuild()

    def _rebuild(self):
        if self._boxes:
            self.xyxy = torch.cat([b.xyxy for b in self._boxes], dim=0)
            self.conf = torch.cat([b.conf for b in self._boxes], dim=0)
            self.cls = torch.cat([b.cls for b in self._boxes], dim=0)
        else:
            self.xyxy = torch.zeros((0, 4), device=self._device)
            self.conf = torch.zeros((0,), device=self._device)
            self.cls = torch.zeros((0,), device=self._device)
        # `data` is what `_filter_predictions` queries for device info.
        self.data = self.xyxy

    def __iter__(self):
        return iter(self._boxes)

    def __len__(self):
        return len(self._boxes)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return _FakeBoxes(self._boxes[item], self._device)
        if isinstance(item, torch.Tensor):
            idxs = item.tolist()
            return _FakeBoxes([self._boxes[i] for i in idxs], self._device)
        if isinstance(item, (list, tuple)):
            return _FakeBoxes([self._boxes[i] for i in item], self._device)
        return self._boxes[item]


class _FakeResult:
    def __init__(self, boxes_list, names, device):
        self.boxes = _FakeBoxes(boxes_list, device) if boxes_list else _FakeBoxes([], device)
        self.names = names
        self.orig_shape = None


def _nms_per_class(detections, iou_thresh):
    """detections: list of (xyxy, conf, cls). Returns a per-class-NMS filtered list."""
    if not detections:
        return []
    by_cls = {}
    for det in detections:
        by_cls.setdefault(det[2], []).append(det)
    keep = []
    for _cls, dets in by_cls.items():
        boxes = torch.tensor([d[0] for d in dets], dtype=torch.float32)
        scores = torch.tensor([d[1] for d in dets], dtype=torch.float32)
        idx = torchvision.ops.nms(boxes, scores, iou_thresh)
        for i in idx.tolist():
            keep.append(dets[i])
    return keep


def derive_lane_region(lane_polygons_buffered, frame_w, frame_h,
                       y_top_margin=40, bottom_fraction=0.78):
    """Compute a lane-aware tile region from the lane polygon union.

    The S20 tested region (0, 200, 640, 380) on 640x480 was empirically the
    upper portion of the lane union — tiling the bottom of near-lane creates
    duplicate detections that NMS can't always merge. We replicate that
    heuristic generically:
      - x: full frame width (vehicles often extend beyond lane polygons)
      - y_top: above the highest lane vertex with a small margin
      - y_bot: ~78% down the lane vertical span (trim the near-lane bottom)
    """
    if not lane_polygons_buffered:
        return (0, 0, frame_w, frame_h)
    ys_min, ys_max = [], []
    for _, poly in lane_polygons_buffered:
        minx, miny, maxx, maxy = poly.bounds
        ys_min.append(miny)
        ys_max.append(maxy)
    lane_y_min = min(ys_min)
    lane_y_max = max(ys_max)
    y_top = max(0, int(lane_y_min) - y_top_margin)
    y_bot = min(frame_h, int(lane_y_min + (lane_y_max - lane_y_min) * bottom_fraction))
    return (0, y_top, frame_w, y_bot)


def sliced_predict(
    model,
    frame,
    conf_thresh=0.03,
    tile_w=320,
    tile_h=240,
    overlap=0.25,
    imgsz_tile=1280,
    imgsz_full=1408,
    include_full=True,
    nms_iou=0.3,
    augment_tile=False,
    augment_full=True,
    lane_aware_only=None,
):
    """SAHI-style sliced inference returning ultralytics-shaped results.

    Defaults match the S20 winning configuration from the validation sweep.

    Args:
        model: ultralytics YOLO model.
        frame: numpy HxWx3 BGR.
        conf_thresh: detector confidence threshold (passed to model.predict).
        tile_w, tile_h: tile dimensions in source-frame pixels.
        overlap: fraction (0-1) of tile dimension to overlap.
        imgsz_tile: inference imgsz for tiles.
        imgsz_full: inference imgsz for the full frame.
        include_full: also run a full-frame inference pass.
        nms_iou: per-class NMS IoU threshold for merging tile + full detections.
        augment_tile: TTA on tiles (S20 = False).
        augment_full: TTA on the full-frame pass (S20 = True).
        lane_aware_only: (x0, y0, x1, y1) rectangle to restrict tiling to.
            None = tile the full frame.

    Returns:
        list with one _FakeResult element compatible with ultralytics consumers.
    """
    h, w = frame.shape[:2]
    step_x = max(1, int(tile_w * (1 - overlap)))
    step_y = max(1, int(tile_h * (1 - overlap)))

    if lane_aware_only is not None:
        rx0, ry0, rx1, ry1 = lane_aware_only
        rx0 = max(0, rx0)
        ry0 = max(0, ry0)
        rx1 = min(w, rx1)
        ry1 = min(h, ry1)
    else:
        rx0, ry0, rx1, ry1 = 0, 0, w, h

    tile_origins = []
    y0 = ry0
    while y0 < ry1:
        x0 = rx0
        while x0 < rx1:
            tile_origins.append((x0, y0))
            x0 += step_x
            if x0 + tile_w > rx1 and x0 - step_x + tile_w < rx1:
                x0 = rx1 - tile_w
        y0 += step_y
        if y0 + tile_h > ry1 and y0 - step_y + tile_h < ry1:
            y0 = ry1 - tile_h

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_dets = []

    for (x0, y0) in tile_origins:
        x1 = min(w, x0 + tile_w)
        y1 = min(h, y0 + tile_h)
        tile = frame[y0:y1, x0:x1]
        if tile.size == 0:
            continue
        results = model.predict(
            tile,
            conf=conf_thresh,
            agnostic_nms=False,
            verbose=False,
            imgsz=imgsz_tile,
            augment=augment_tile,
        )
        for r in results:
            if r.boxes is None:
                continue
            for b in r.boxes:
                bx = b.xyxy[0].cpu().numpy()
                all_dets.append((
                    (float(bx[0] + x0), float(bx[1] + y0),
                     float(bx[2] + x0), float(bx[3] + y0)),
                    float(b.conf.item()),
                    int(b.cls.item()),
                ))

    if include_full:
        full = model.predict(
            frame,
            conf=conf_thresh,
            agnostic_nms=False,
            verbose=False,
            imgsz=imgsz_full,
            augment=augment_full,
        )
        for r in full:
            if r.boxes is None:
                continue
            for b in r.boxes:
                bx = b.xyxy[0].cpu().numpy()
                all_dets.append((
                    (float(bx[0]), float(bx[1]), float(bx[2]), float(bx[3])),
                    float(b.conf.item()),
                    int(b.cls.item()),
                ))

    merged = _nms_per_class(all_dets, iou_thresh=nms_iou)
    fake_boxes = [_FakeBox(*d, device=device) for d in merged]
    res = _FakeResult(fake_boxes, model.names, device)
    res.orig_shape = (h, w)
    return [res]
