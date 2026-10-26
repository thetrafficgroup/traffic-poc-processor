import json, math, os

ROOT = "/Users/nacho/dev/genium/traffic-v2/proc-iter-tmc"
SITES = json.load(open(f"{ROOT}/round1_out/sites.json"))
K = 0.5  # zone depth as a fraction of the leg line's own length


def cfg_path(rel):
    for base in (".", "gt", "round1_out"):
        p = os.path.join(ROOT, base, rel)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(rel)


def quad(p1, p2, depth, ux, uy):
    return [p1, p2, (p2[0] + depth * ux, p2[1] + depth * uy),
            (p1[0] + depth * ux, p1[1] + depth * uy)]


def area(poly):
    n = len(poly)
    return abs(sum(poly[i][0] * poly[(i + 1) % n][1] - poly[(i + 1) % n][0] * poly[i][1]
                   for i in range(n))) / 2


def clip(poly, plane):
    """Sutherland-Hodgman against a half-plane given as (a, b, c): ax+by+c >= 0."""
    a, b, c = plane
    out = []
    n = len(poly)
    for i in range(n):
        cur, nxt = poly[i], poly[(i + 1) % n]
        dc, dn = a * cur[0] + b * cur[1] + c, a * nxt[0] + b * nxt[1] + c
        if dc >= 0:
            out.append(cur)
        if (dc >= 0) != (dn >= 0):
            t = dc / (dc - dn)
            out.append((cur[0] + t * (nxt[0] - cur[0]), cur[1] + t * (nxt[1] - cur[1])))
    return out


def planes_of(poly):
    """Inward half-planes of a convex polygon wound either way."""
    n = len(poly)
    sgn = 1.0
    s = sum(poly[i][0] * poly[(i + 1) % n][1] - poly[(i + 1) % n][0] * poly[i][1]
            for i in range(n))
    if s < 0:
        sgn = -1.0
    out = []
    for i in range(n):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % n]
        a, b = -sgn * (y2 - y1), sgn * (x2 - x1)
        out.append((a, b, -(a * x1 + b * y1)))
    return out


def inter_area(p, q):
    poly = p
    for pl in planes_of(q):
        poly = clip(poly, pl)
        if len(poly) < 3:
            return 0.0
    return area(poly)


print(f"{'site':20} {'legs':5} {'skew max':9} {'overlap%':9} {'offframe% max':13} worst")
flag = []
for row in SITES:
    d = json.load(open(cfg_path(row["lines"])))
    W, H = (int(v) for v in row["res"].split("x"))
    legs = {k: v for k, v in d.items() if k != "crosswalks"}
    def ends(v):
        q = v.get("points") or [v["pt1"], v["pt2"]]
        return tuple(q[0]), tuple(q[-1])
    pts = [p for v in legs.values() for p in ends(v)]
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)

    zones, skews = {}, {}
    for name, v in legs.items():
        p1, p2 = ends(v)
        mid = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
        L = math.dist(p1, p2)
        # outward = from the intersection centre towards this leg's line
        ox, oy = mid[0] - cx, mid[1] - cy
        on = math.hypot(ox, oy) or 1.0
        ox, oy = ox / on, oy / on
        # the line's own normal, signed outward
        nx, ny = -(p2[1] - p1[1]) / L, (p2[0] - p1[0]) / L
        if nx * ox + ny * oy < 0:
            nx, ny = -nx, -ny
        skews[name] = math.degrees(math.acos(max(-1, min(1, nx * ox + ny * oy))))
        zones[name] = quad(p1, p2, K * L, nx, ny)

    frame = [(0, 0), (W, 0), (W, H), (0, H)]
    off = {n: 1 - inter_area(z, frame) / area(z) for n, z in zones.items()}
    names = list(zones)
    worst_ov, worst_pair = 0.0, ""
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = zones[names[i]], zones[names[j]]
            ov = inter_area(a, b) / min(area(a), area(b))
            if ov > worst_ov:
                worst_ov, worst_pair = ov, f"{names[i]}/{names[j]}"
    ms = max(skews.values())
    mo = max(off.values())
    note = []
    if ms > 25: note.append(f"skew {max(skews, key=skews.get)}")
    if worst_ov > 0.05: note.append(f"overlap {worst_pair}")
    if mo > 0.25: note.append(f"offframe {max(off, key=off.get)}")
    if note: flag.append(row["site"])
    print(f"{row['site']:20} {len(legs):<5} {ms:7.1f}°  {worst_ov*100:7.1f}%  "
          f"{mo*100:11.1f}%   {', '.join(note)}")

print(f"\nflagged {len(flag)}/{len(SITES)}: {', '.join(flag) if flag else 'none'}")
