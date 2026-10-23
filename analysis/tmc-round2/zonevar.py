import json, math, os
exec(open("zoneaudit.py").read().split('print(f"{\'site\':20}')[0])  # reuse helpers

def build(p1, p2, nx, ny, L, k, mode):
    if mode == "outward":  return quad(p1, p2, k * L, nx, ny)
    if mode == "inward":   return quad(p1, p2, k * L, -nx, -ny)
    d = k * L / 2                                    # straddling band
    b1 = (p1[0] - d * nx, p1[1] - d * ny)
    b2 = (p2[0] - d * nx, p2[1] - d * ny)
    return quad(b1, b2, k * L, nx, ny)

for mode in ("outward", "inward", "band"):
    for K in (0.3, 0.5):
        ovs, offs, worst = [], [], []
        for row in SITES:
            d = json.load(open(cfg_path(row["lines"])))
            W, H = (int(v) for v in row["res"].split("x"))
            legs = {k: v for k, v in d.items() if k != "crosswalks"}
            def ends(v):
                q = v.get("points") or [v["pt1"], v["pt2"]]
                return tuple(q[0]), tuple(q[-1])
            pts = [p for v in legs.values() for p in ends(v)]
            cx = sum(p[0] for p in pts) / len(pts); cy = sum(p[1] for p in pts) / len(pts)
            zones = {}
            for name, v in legs.items():
                p1, p2 = ends(v)
                L = math.dist(p1, p2)
                mid = ((p1[0]+p2[0])/2, (p1[1]+p2[1])/2)
                ox, oy = mid[0]-cx, mid[1]-cy
                nx, ny = -(p2[1]-p1[1])/L, (p2[0]-p1[0])/L
                if nx*ox + ny*oy < 0: nx, ny = -nx, -ny
                zones[name] = build(p1, p2, nx, ny, L, K, mode)
            frame = [(0,0),(W,0),(W,H),(0,H)]
            for n, z in zones.items():
                offs.append(1 - inter_area(z, frame)/area(z))
            names = list(zones)
            wv = 0
            for i in range(len(names)):
                for j in range(i+1, len(names)):
                    a, b = zones[names[i]], zones[names[j]]
                    ov = inter_area(a, b)/min(area(a), area(b))
                    ovs.append(ov); wv = max(wv, ov)
            worst.append((wv, row["site"]))
        offs.sort(); worst.sort(reverse=True)
        bad_off = sum(1 for o in offs if o > 0.4)
        print(f"{mode:8} K={K}  overlap: mean {sum(ovs)/len(ovs)*100:5.1f}% max {max(ovs)*100:5.1f}% ({worst[0][1]})"
              f"   offframe: median {offs[len(offs)//2]*100:5.1f}% >40%: {bad_off}/{len(offs)} legs")
