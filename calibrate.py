#!/usr/bin/env python3
"""
calibrate.py — Lane grid calibration from fixed CCTV footage.
Saves grid_config_cam_01.json.

Run:  python calibrate.py
Each checkpoint shows a debug image. SPACE/ENTER continues, Q quits.
"""
import json, os, sys
import cv2
import numpy as np

# ─── Config ──────────────────────────────────────────────────────
VIDEO_PATH  = "in/car_100kmh.mp4"
MOCK_PATH   = "mock_tracks.json"
H_PATH      = "H_manual.npy"
OUT_JSON    = "grid_config_cam_01.json"

SAMPLE_EVERY = 15
MAX_SAMPLE   = 300
YOLO_EVERY   = 3
MAX_YOLO     = 150

HOUGH_RHO      = 1
HOUGH_THETA    = np.pi / 180
HOUGH_THRESH   = 30
HOUGH_MIN_LEN  = 60
HOUGH_MAX_GAP  = 20

ANGLE_DIST_PX    = 200
ANGLE_DIFF_DEG   = 12
POLY_DEG         = 2     # polynomial degree for lane curve fitting
CURVE_N          = 400   # sample points along each fitted curve
CLUSTER_EPS      = 60    # DBSCAN: max distance in combined (position + angle) space
CLUSTER_MIN_SEGS = 3     # DBSCAN: min segments to form a cluster
CLUSTER_ANG_W      = 200   # how much 1 radian of angle diff counts (px equivalent)
                           # rule of thumb: ~12° diff → 2*sin(12°)*200 ≈ 83 px → separate cluster
OUTLIER_SIZE_RATIO = 0.10  # cluster flagged if seg count < max_count * ratio


# ─── Helpers ─────────────────────────────────────────────────────
def _track_color(tid):
    hue = int((tid * 137) % 180)
    px = np.uint8([[[hue, 220, 220]]])
    return cv2.cvtColor(px, cv2.COLOR_HSV2BGR)[0, 0].tolist()


def checkpoint(name, img):
    """Write debug PNG, show window, block until SPACE/ENTER or Q."""
    path = f"debug_{name}.png"
    cv2.imwrite(path, img)
    print(f"\n[checkpoint] {path}  →  SPACE/ENTER=continue  Q=quit")
    win = f"[{name}]"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)
    cv2.imshow(win, img)
    while True:
        k = cv2.waitKey(0) & 0xFF
        if k in (32, 13, 10):   # SPACE or ENTER
            break
        if k == ord('q'):
            cv2.destroyAllWindows()
            print("Quit.")
            sys.exit(0)
    cv2.destroyWindow(win)
    cv2.waitKey(1)


def color_filter(frame, k=3, blur_ksize=31):
    """Keep dominant-color pixels intact; blur the rest (suppresses trees/sky)."""
    small = cv2.resize(frame, (frame.shape[1] // 4, frame.shape[0] // 4))
    pixels = small.reshape(-1, 3).astype(np.float32)
    _, labels, _ = cv2.kmeans(
        pixels, k, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
        3, cv2.KMEANS_RANDOM_CENTERS,
    )
    counts = np.bincount(labels.flatten(), minlength=k)
    dominant = int(np.argmax(counts))
    lab_small = labels.reshape(small.shape[:2]).astype(np.uint8)
    lab_full = cv2.resize(lab_small, (frame.shape[1], frame.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    blurred = cv2.GaussianBlur(frame, (blur_ksize, blur_ksize), 0)
    result = blurred.copy()
    result[lab_full == dominant] = frame[lab_full == dominant]
    return result


def preprocess(frame):
    """CLAHE → Gaussian blur → Canny → clip top 5%."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edges[: int(edges.shape[0] * 0.05)] = 0
    return edges


def hsv_lane_mask(frame):
    """White and yellow lane marking mask."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white  = cv2.inRange(hsv, (0,  0,  180), (180, 30,  255))
    yellow = cv2.inRange(hsv, (15, 80,  80), ( 35, 255, 255))
    return cv2.bitwise_or(white, yellow)


def seg_angle(x1, y1, x2, y2):
    return np.degrees(np.arctan2(y2 - y1, x2 - x1))


def seg_mid(x1, y1, x2, y2):
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def poly_fit(segs, deg=POLY_DEG):
    """Collect seg endpoints, fit x = poly(y). Returns (poly1d, y_min, y_max) or None."""
    pts = sorted(
        [(x, y) for s in segs for x, y in [(s[0], s[1]), (s[2], s[3])]],
        key=lambda p: p[1],
    )
    if len(pts) < deg + 2:
        return None
    ys = np.array([p[1] for p in pts], dtype=np.float64)
    xs = np.array([p[0] for p in pts], dtype=np.float64)
    _, idx = np.unique(ys, return_index=True)
    ys, xs = ys[idx], xs[idx]
    if len(ys) < deg + 2:
        return None
    coeffs = np.polyfit(ys, xs, deg=min(deg, len(ys) - 1))
    return np.poly1d(coeffs), float(ys.min()), float(ys.max())


def eval_curve(fit, n=CURVE_N):
    """Evaluate poly over its y range → (N,1,2) int32 for cv2.polylines, or empty."""
    if fit is None:
        return np.zeros((0, 1, 2), dtype=np.int32)
    poly, y_min, y_max = fit
    ys = np.linspace(y_min, y_max, n)
    xs = poly(ys)
    ok = (xs >= 0) & (xs < frame_W) & (ys >= 0) & (ys < frame_H)
    return np.column_stack([xs[ok], ys[ok]]).astype(np.int32).reshape(-1, 1, 2)


def unproject_curve(curve_px, H):
    """Image (N,1,2) → world (N,2) [X, Z] via ground-plane homography."""
    if len(curve_px) == 0 or H is None:
        return np.zeros((0, 2))
    return cv2.perspectiveTransform(curve_px.astype(np.float32), H).reshape(-1, 2)


# ═══════════════════════════════════════════════════════════════════
# Stage 0 — Sample frames + car traces
# ═══════════════════════════════════════════════════════════════════
print("=== Stage 0: sample frames + car traces ===")

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise SystemExit(f"Cannot open {VIDEO_PATH}")

frame_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

sampled = []
fi = 0
while len(sampled) < MAX_SAMPLE:
    ret, fr = cap.read()
    if not ret:
        break
    if fi % SAMPLE_EVERY == 0:
        sampled.append(fr.copy())
    fi += 1
cap.release()

if not sampled:
    raise SystemExit("No frames sampled — check VIDEO_PATH")
print(f"  {len(sampled)} frames sampled  ({frame_W}×{frame_H})")

ref = sampled[len(sampled) // 2]

if os.path.exists(MOCK_PATH):
    with open(MOCK_PATH) as f:
        raw = json.load(f)
    tracks = {int(k): [tuple(p) for p in v] for k, v in raw.items()}
    print(f"  mock mode: {len(tracks)} tracks from {MOCK_PATH}")
else:
    print("  YOLO mode — running YOLOv11 + ByteTrack …")
    from ultralytics import YOLO  # noqa: E402
    yolo = YOLO("yolo11m.pt")
    tracks: dict = {}
    cap2 = cv2.VideoCapture(VIDEO_PATH)
    fi = yolo_n = 0
    while yolo_n < MAX_YOLO:
        ret, fr = cap2.read()
        if not ret:
            break
        if fi % YOLO_EVERY == 0:
            res = yolo.track(fr, classes=[2, 3, 5, 7], persist=True,
                             tracker="bytetrack.yaml", verbose=False)
            boxes = res[0].boxes
            if boxes.id is not None:
                for box, tid in zip(boxes, boxes.id.int().tolist()):
                    x1b, y1b, x2b, y2b = map(int, box.xyxy[0])
                    tracks.setdefault(tid, []).append(((x1b + x2b) // 2, y2b))
            yolo_n += 1
        fi += 1
    cap2.release()
    print(f"  YOLO: {len(tracks)} tracks")

# SVD → dominant road direction
disps = [
    [pts[i][0] - pts[i-1][0], pts[i][1] - pts[i-1][1]]
    for pts in tracks.values()
    for i in range(1, len(pts))
    if (pts[i][0]-pts[i-1][0])**2 + (pts[i][1]-pts[i-1][1])**2 > 9
]
road_angle = 90.0
if len(disps) >= 2:
    _, _, Vt = np.linalg.svd(np.array(disps, dtype=np.float64), full_matrices=False)
    a = np.degrees(np.arctan2(Vt[0, 1], Vt[0, 0]))
    road_angle = a - 180 if a > 90 else (a + 180 if a < -90 else a)
print(f"  road_angle = {road_angle:.1f}°")

dbg0 = ref.copy()
for tid, pts in tracks.items():
    col = _track_color(tid)
    for j in range(1, len(pts)):
        cv2.line(dbg0, pts[j-1], pts[j], col, 2, cv2.LINE_AA)
    if pts:
        cv2.circle(dbg0, pts[-1], 5, col, -1)
        cv2.putText(dbg0, str(tid), (pts[-1][0]+6, pts[-1][1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
cv2.putText(dbg0, f"tracks={len(tracks)}  road_angle={road_angle:.1f}deg",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
checkpoint("0_car_traces", dbg0)


# ═══════════════════════════════════════════════════════════════════
# Stage 1 — Color filter (side-by-side)
# ═══════════════════════════════════════════════════════════════════
print("=== Stage 1: color filter ===")
filt_ref = color_filter(ref)
dbg1 = np.hstack([ref, filt_ref])
cv2.putText(dbg1, "original",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
cv2.putText(dbg1, "color_filtered",
            (ref.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
checkpoint("1_color_filter", dbg1)


# ═══════════════════════════════════════════════════════════════════
# Stage 2 — Edges (preprocess + hsv_lane_mask)
# ═══════════════════════════════════════════════════════════════════
print("=== Stage 2: edges ===")
edges_raw = preprocess(filt_ref)
lmask_ref = hsv_lane_mask(ref)
edges_masked = cv2.bitwise_and(edges_raw, lmask_ref)

# 3 independent panels so layers don't overwrite each other
p1 = cv2.cvtColor(edges_raw, cv2.COLOR_GRAY2BGR)          # raw Canny (white on black)

p2 = np.zeros_like(ref)
p2[lmask_ref > 0] = (0, 200, 0)                           # green — HSV white/yellow mask

p3 = np.zeros_like(ref)
p3[edges_masked > 0] = (0, 255, 255)                      # yellow — Hough input

for text, panel in [
    ("1) Canny edges", p1),
    ("2) HSV lane mask (white+yellow)", p2),
    ("3) result = edges & mask  [Hough input]", p3),
]:
    cv2.putText(panel, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 0), 2)

dbg2 = np.hstack([p1, p2, p3])
checkpoint("2_edges", dbg2)


# ═══════════════════════════════════════════════════════════════════
# Stage 3a — Hough on all frames + trace-angle filter
# ═══════════════════════════════════════════════════════════════════
print("=== Stage 3a: Hough accumulation + angle filter ===")

# Build trace micro-segments with their angles
trace_segs = []
for pts in tracks.values():
    for i in range(1, len(pts)):
        x1t, y1t = pts[i-1]
        x2t, y2t = pts[i]
        trace_segs.append((x1t, y1t, x2t, y2t,
                           seg_angle(x1t, y1t, x2t, y2t)))

all_segs = []
for n, fr in enumerate(sampled):
    filt = color_filter(fr)
    edges = preprocess(filt)
    em = cv2.bitwise_and(edges, hsv_lane_mask(fr))
    lines = cv2.HoughLinesP(em, HOUGH_RHO, HOUGH_THETA, HOUGH_THRESH,
                             minLineLength=HOUGH_MIN_LEN, maxLineGap=HOUGH_MAX_GAP)
    if lines is not None:
        for ln in lines:
            all_segs.append(tuple(map(int, ln[0])))
    if (n + 1) % 20 == 0:
        print(f"  Hough: {n+1}/{len(sampled)} frames …")

print(f"  {len(all_segs)} raw segments total")

kept, rejected, orphans = [], [], []
for seg in all_segs:
    mx, my = seg_mid(*seg)
    sa = seg_angle(*seg)

    # find closest trace micro-segment
    best_dist = float('inf')
    best_ta = 0.0
    for tx1, ty1, tx2, ty2, ta in trace_segs:
        d = np.hypot(mx - (tx1 + tx2) / 2, my - (ty1 + ty2) / 2)
        if d < best_dist:
            best_dist = d
            best_ta = ta

    if best_dist > ANGLE_DIST_PX:
        orphans.append(seg)
    else:
        adiff = abs(sa - best_ta)
        adiff = min(adiff, 180 - adiff)
        (kept if adiff <= ANGLE_DIFF_DEG else rejected).append(seg)

print(f"  kept={len(kept)}  rejected={len(rejected)}  orphan={len(orphans)}")

dbg3a = ref.copy()
for seg in rejected: cv2.line(dbg3a, seg[:2], seg[2:], (0,   0, 200), 1)
for seg in orphans:  cv2.line(dbg3a, seg[:2], seg[2:], (0, 140, 255), 1)
for seg in kept:     cv2.line(dbg3a, seg[:2], seg[2:], (0, 220,   0), 1)
for tid, pts in tracks.items():
    col = _track_color(tid)
    for j in range(1, len(pts)):
        cv2.line(dbg3a, pts[j-1], pts[j], col, 2, cv2.LINE_AA)
cv2.putText(dbg3a,
            f"green=kept({len(kept)})  red=rej({len(rejected)})  orange=orphan({len(orphans)})",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
checkpoint("3_angle_filter", dbg3a)


# ═══════════════════════════════════════════════════════════════════
# Stage 4 — Green lines (clean display of kept segments only)
# ═══════════════════════════════════════════════════════════════════
print("=== Stage 4: green lines ===")

if not kept:
    raise SystemExit("No kept segments — check ANGLE_DIFF_DEG / ANGLE_DIST_PX config")

dbg4 = ref.copy()
for seg in kept:
    cv2.line(dbg4, seg[:2], seg[2:], (0, 255, 0), 2, cv2.LINE_AA)
cv2.putText(dbg4, f"kept (green) segments: {len(kept)}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
checkpoint("4_green_lines", dbg4)


# ── DBSCAN: cluster by position + angle ──────────────────────────
# Feature per segment: [x_mid, y_mid, cos(2θ)·W, sin(2θ)·W]
# cos/sin of 2θ handles 180° symmetry (same line, opposite direction = same cluster).
# W scales angle so that ~12° difference ≈ 80 px in feature space → separate cluster.
from sklearn.cluster import DBSCAN

def _seg_feature(seg):
    mx, my = seg_mid(*seg)
    a = np.radians(seg_angle(*seg))
    return [mx, my, np.cos(2 * a) * CLUSTER_ANG_W, np.sin(2 * a) * CLUSTER_ANG_W]

feats = np.array([_seg_feature(s) for s in kept], dtype=np.float32)
db_labels = DBSCAN(eps=CLUSTER_EPS, min_samples=CLUSTER_MIN_SEGS).fit(feats).labels_

clusters = {}   # label → [seg, ...]
noise_segs = []
for seg, lbl in zip(kept, db_labels):
    if lbl == -1:
        noise_segs.append(seg)
    else:
        clusters.setdefault(lbl, []).append(seg)

print(f"  DBSCAN → {len(clusters)} clusters  {len(noise_segs)} noise segs")

# Flag outlier clusters whose seg count is suspiciously small
if clusters:
    max_size = max(len(s) for s in clusters.values())
    outlier_set = {lbl for lbl, segs in clusters.items()
                   if len(segs) < max_size * OUTLIER_SIZE_RATIO}
else:
    outlier_set = set()

valid_clusters = {lbl: segs for lbl, segs in clusters.items() if lbl not in outlier_set}
print(f"  outliers: {sorted(outlier_set)}  valid: {sorted(valid_clusters)}")


# ═══════════════════════════════════════════════════════════════════
# Stage 4b — One straight line per cluster
# ═══════════════════════════════════════════════════════════════════
print("=== Stage 4b: straight line per cluster ===")

dbg4b = ref.copy()
# DBSCAN noise in dark grey
for seg in noise_segs:
    cv2.line(dbg4b, seg[:2], seg[2:], (60, 60, 60), 1)

for lbl, segs in clusters.items():
    is_outlier = lbl in outlier_set
    col = (0, 0, 220) if is_outlier else _track_color(lbl)   # red if outlier
    dim = [c // 2 for c in col]
    for seg in segs:
        cv2.line(dbg4b, seg[:2], seg[2:], dim, 1)
    fit1 = poly_fit(segs, deg=1)
    if fit1 is None:
        continue
    p1, _, _ = fit1
    pt_top = (int(np.clip(p1(0),           0, frame_W - 1)), 0)
    pt_bot = (int(np.clip(p1(frame_H - 1), 0, frame_W - 1)), frame_H - 1)
    cv2.line(dbg4b, pt_top, pt_bot, col, 2, cv2.LINE_AA)
    tag = f"C{lbl}({len(segs)}) NOISE" if is_outlier else f"C{lbl}({len(segs)})"
    lx = min(pt_bot[0] + 6, frame_W - 90)
    cv2.putText(dbg4b, tag, (lx, pt_bot[1] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

n_valid = len(valid_clusters)
n_out   = len(outlier_set)
cv2.putText(dbg4b,
            f"{n_valid} valid (colored)  {n_out} outlier (red)  ratio={OUTLIER_SIZE_RATIO}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
checkpoint("4b_straight_lines", dbg4b)


# ═══════════════════════════════════════════════════════════════════
# Stage 5 — 2D Polynomial fitting per cluster
# ═══════════════════════════════════════════════════════════════════
print("=== Stage 5: 2D polynomial fitting ===")

fits   = {lbl: poly_fit(segs) for lbl, segs in valid_clusters.items()}
curves = {lbl: eval_curve(fit) for lbl, fit in fits.items()}

dbg5 = ref.copy()
for seg in kept:
    cv2.line(dbg5, seg[:2], seg[2:], (0, 100, 0), 1)    # dim green
# outlier clusters: dim red dots, no curve
for lbl, segs in clusters.items():
    if lbl not in outlier_set:
        continue
    for seg in segs:
        for pt in [seg[:2], seg[2:]]:
            cv2.circle(dbg5, pt, 2, (0, 0, 120), -1)

# valid clusters: colored dots + polynomial curve
for lbl, segs in valid_clusters.items():
    col = _track_color(lbl)
    for seg in segs:
        for pt in [seg[:2], seg[2:]]:
            cv2.circle(dbg5, pt, 2, col, -1)
    crv = curves[lbl]
    if len(crv) > 1:
        cv2.polylines(dbg5, [crv], False, col, 2, cv2.LINE_AA)

cv2.putText(dbg5,
            f"{len(valid_clusters)} poly curves  (deg={POLY_DEG})  {len(outlier_set)} outlier skipped",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
checkpoint("5_spline", dbg5)


# ═══════════════════════════════════════════════════════════════════
# Stage 6 — Continuous curves (clean, no raw segments)
# ═══════════════════════════════════════════════════════════════════
print("=== Stage 6: continuous curves ===")

dbg6 = ref.copy()
for lbl, crv in curves.items():
    if len(crv) < 2:
        continue
    col = _track_color(lbl)
    cv2.polylines(dbg6, [crv], False, col, 3, cv2.LINE_AA)
    for pt in crv[::20, 0]:
        cv2.circle(dbg6, tuple(pt), 3, col, -1)

cv2.putText(dbg6, f"{len(curves)} continuous polynomial curves",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
checkpoint("6_curves", dbg6)


# Load homography once here — reused by Stage 6b and Stage 7
H_gnd = np.load(H_PATH) if os.path.exists(H_PATH) else None
H_inv = np.linalg.inv(H_gnd) if H_gnd is not None else None


# ═══════════════════════════════════════════════════════════════════
# Stage 6b — BEV polynomial fit (project to world → fit → warp back)
# ═══════════════════════════════════════════════════════════════════
print("=== Stage 6b: BEV deg-2 polynomial fit ===")

BEV_DEG    = 2   # deg-1 or 2 is stable in rectified BEV space
global_bev_crv = np.zeros((0, 1, 2), dtype=np.int32)
n_removed_bev  = 0

if H_gnd is None:
    print(f"  {H_PATH} not found — skipping BEV fit (run manual_calibrate.py)")
else:
    # Project per-cluster curve points (debug_6) into world (BEV) coordinates
    # curves[lbl] shape (N,1,2) → perspectiveTransform → (N,2) world [X, Z]
    bev_pts = []
    for crv in curves.values():
        if len(crv) == 0:
            continue
        w = cv2.perspectiveTransform(crv.astype(np.float32), H_gnd).reshape(-1, 2)
        bev_pts.extend(zip(w[:, 0].tolist(), w[:, 1].tolist()))  # (X_world, Z_world)

    bev_pts.sort(key=lambda p: p[1])   # sort by depth Z
    Zs = np.array([p[1] for p in bev_pts], dtype=np.float64)
    Xs = np.array([p[0] for p in bev_pts], dtype=np.float64)

    # Deduplicate Z
    _, idx = np.unique(Zs, return_index=True)
    Zs, Xs = Zs[idx], Xs[idx]

    if len(Zs) >= BEV_DEG + 2:
        # Pass 1: initial fit → residual-based outlier rejection
        c1_bev = np.polyfit(Zs, Xs, deg=BEV_DEG)
        res_bev = np.abs(Xs - np.poly1d(c1_bev)(Zs))
        thresh_bev = np.median(res_bev) + 2.0 * res_bev.std()
        keep_bev = res_bev <= thresh_bev
        n_removed_bev = int((~keep_bev).sum())
        Zs_f, Xs_f = Zs[keep_bev], Xs[keep_bev]

        if len(Zs_f) >= BEV_DEG + 2:
            # Pass 2: clean fit in BEV
            c_bev = np.polyfit(Zs_f, Xs_f, deg=BEV_DEG)
            poly_bev = np.poly1d(c_bev)

            # Evaluate in world space, warp back to image
            Z_ev = np.linspace(float(Zs_f.min()), float(Zs_f.max()), CURVE_N)
            X_ev = poly_bev(Z_ev)
            world_ev = np.column_stack([X_ev, Z_ev]).astype(np.float32).reshape(-1, 1, 2)
            img_ev = cv2.perspectiveTransform(world_ev, H_inv).reshape(-1, 2)
            ok = ((img_ev[:, 0] >= 0) & (img_ev[:, 0] < frame_W) &
                  (img_ev[:, 1] >= 0) & (img_ev[:, 1] < frame_H))
            global_bev_crv = img_ev[ok].astype(np.int32).reshape(-1, 1, 2)

    print(f"  BEV pts={len(Zs)}  outliers_removed={n_removed_bev}  pts_used={len(Zs) - n_removed_bev}")

dbg6b = ref.copy()
for lbl, crv in curves.items():
    if len(crv) < 2:
        continue
    cv2.polylines(dbg6b, [crv], False, _track_color(lbl), 2, cv2.LINE_AA)

if len(global_bev_crv) > 1:
    cv2.polylines(dbg6b, [global_bev_crv], False, (255, 255, 255), 3, cv2.LINE_AA)
    bpt = tuple(global_bev_crv[-1, 0])
    cv2.putText(dbg6b, "BEV deg-2", (bpt[0] + 8, bpt[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

cv2.putText(dbg6b,
            f"BEV deg-{BEV_DEG} (white)  fit in world-space  outliers_removed={n_removed_bev}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
checkpoint("6b_global_fit", dbg6b)


# ═══════════════════════════════════════════════════════════════════
# Stage 7 — 3D grid overlay on image
# ═══════════════════════════════════════════════════════════════════
print("=== Stage 7: 3D grid overlay ===")

GRID_COLS = 5   # lateral divisions between lane boundaries
GRID_ROWS = 8   # depth divisions along road

# H_gnd and H_inv already loaded before Stage 6b
world_curves: dict = {}
world_grid = None   # filled by Stage 7; used by Stage 8

if H_gnd is None:
    print(f"  {H_PATH} not found — run manual_calibrate.py; skipping grid overlay")
else:
    world_curves = {lbl: unproject_curve(crv, H_gnd) for lbl, crv in curves.items()}

    # Sort curves left → right by mean world-X
    sorted_lbls = sorted(world_curves,
                         key=lambda l: world_curves[l][:, 0].mean()
                         if len(world_curves[l]) > 0 else 0)

    if len(sorted_lbls) == 0:
        print("  No valid world curves — cannot build grid")
    else:
        if len(sorted_lbls) >= 2:
            w_left  = world_curves[sorted_lbls[0]]
            w_right = world_curves[sorted_lbls[-1]]
        else:
            w_c = world_curves[sorted_lbls[0]]
            w_left  = w_c.copy(); w_left[:,  0] -= 3.5
            w_right = w_c.copy(); w_right[:, 0] += 3.5

        all_z = np.concatenate([w_left[:, 1], w_right[:, 1]])
        z_min, z_max = float(all_z.min()), float(all_z.max())

        def _x_at_z(wpts, z):
            """Interpolate world-X on a (N,2) [X,Z] curve at a given Z depth."""
            order = np.argsort(wpts[:, 1])
            return float(np.interp(z, wpts[order, 1], wpts[order, 0]))

        def _w2px(X, Z):
            """World (X, Z) → image pixel, clamped to frame."""
            pt = cv2.perspectiveTransform(
                np.array([[[X, Z]]], dtype=np.float32), H_inv)[0, 0]
            return (int(np.clip(pt[0], 0, frame_W - 1)),
                    int(np.clip(pt[1], 0, frame_H - 1)))

        # Build grid: (GRID_ROWS+1) depth levels × (GRID_COLS+2) lateral points
        zs = np.linspace(z_min, z_max, GRID_ROWS + 1)
        grid = []
        world_grid = np.zeros((GRID_ROWS + 1, GRID_COLS + 2, 2))
        for ri, z in enumerate(zs):
            xl = _x_at_z(w_left,  z)
            xr = _x_at_z(w_right, z)
            xs_row = np.linspace(xl, xr, GRID_COLS + 2)
            row = [_w2px(x, z) for x in xs_row]
            grid.append(row)
            world_grid[ri, :, 0] = xs_row
            world_grid[ri, :, 1] = z

        dbg7 = ref.copy()
        for row in grid:
            for i in range(len(row) - 1):
                cv2.line(dbg7, row[i], row[i + 1], (0, 200, 255), 1, cv2.LINE_AA)
        for ci in range(GRID_COLS + 2):
            for ri in range(len(grid) - 1):
                cv2.line(dbg7, grid[ri][ci], grid[ri + 1][ci],
                         (0, 255, 120), 1, cv2.LINE_AA)
        for lbl, crv in curves.items():
            if len(crv) > 1:
                cv2.polylines(dbg7, [crv], False, _track_color(lbl), 2, cv2.LINE_AA)

        cv2.putText(dbg7,
                    f"grid {GRID_COLS+2}×{GRID_ROWS+1}  cyan=horiz  green=depth  curves=lane fits",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        checkpoint("7_grid_overlay", dbg7)


# ── Save JSON ────────────────────────────────────────────────────
config: dict = {
    "video": VIDEO_PATH,
    "frame_size": [frame_W, frame_H],
    "road_angle_deg": round(road_angle, 2),
    "cluster_eps": CLUSTER_EPS,
    "curves": {},
}
for lbl, fit in fits.items():
    if fit is None:
        continue
    poly, y_min, y_max = fit
    entry: dict = {
        "n_segs": len(clusters[lbl]),
        "poly_coeffs": poly.coeffs.tolist(),
        "y_range": [float(y_min), float(y_max)],
        "degree": int(len(poly.coeffs) - 1),
    }
    if H_gnd is not None:
        world = world_curves.get(lbl, np.zeros((0, 2)))
        if len(world) > 0:
            entry["world_XZ_samples"] = [
                [round(float(p[0]), 3), round(float(p[1]), 3)]
                for p in world[::10]
            ]
    config["curves"][str(lbl)] = entry

with open(OUT_JSON, "w") as f:
    json.dump(config, f, indent=2)
print(f"Saved {OUT_JSON}")
