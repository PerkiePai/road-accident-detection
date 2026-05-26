import sys
import json
import math
import cv2
import numpy as np

from grid_overlay import draw_grid

VIDEO_PATH        = "in/car_long.mp4"
CAMERA_ID         = "cam_01"
NUM_COLS          = 3
NUM_ROWS          = 8
ROI_TOP_RATIO     = 0.05   # search area: lower 65% of frame, full width
GRID_TOP_RATIO    = 0.40
HOUGH_THRESHOLD   = 50
MIN_LINE_LENGTH   = 50
MAX_LINE_GAP      = 20
ANGLE_MIN_DEG     = 10     # filter out near-horizontal noise
ANGLE_MAX_DEG     = 80     # filter out near-vertical noise
N_LANE_LINES      = 2      # number of lane boundary groups to find
K_COLORS          = 2      # k-means clusters for color filter
BLUR_KERNEL       = 89     # blur strength for non-dominant pixels (must be odd)
FRAME_SAMPLE_STEP = 15     # sample every N frames
MAX_SAMPLE_FRAMES = 300     # cap total frames sampled
DEBUG             = True   # save debug_*.png images each run
YOLO_WEIGHTS      = "yolo11n.pt"
ROAD_ANGLE_TOL    = 12     # ±degrees: how close a marker angle must be to nearby traces
SEARCH_RADIUS     = 200    # pixels: how far a trace segment can be to count as "nearby"
TRACK_FRAME_STEP  = 3      # dense sampling for tracking (keeps ByteTrack IDs alive)
TRACK_MAX_FRAMES  = 150    # cap for tracking pass
DEDUP_DIST        = 20     # pixels: midpoint proximity to call two segments duplicates
DEDUP_ANGLE       = 8      # degrees: angle similarity to call two segments duplicates
WHITE_S_MAX       = 60     # HSV saturation ceiling for white markings
WHITE_V_MIN       = 160    # HSV value floor for white markings
YELLOW_H_MIN      = 15     # HSV hue range for yellow markings
YELLOW_H_MAX      = 35
YELLOW_S_MIN      = 80
YELLOW_V_MIN      = 100


def sample_frames(video_path: str) -> list:
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    idx = 0
    while len(frames) < MAX_SAMPLE_FRAMES and idx < total:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        idx += FRAME_SAMPLE_STEP
    cap.release()
    if not frames:
        sys.exit(f"[calibrate] ERROR: Could not read any frames from {video_path}")
    print(f"[calibrate] Sampled {len(frames)} frames (every {FRAME_SAMPLE_STEP} frames)")
    return frames


def compute_road_angle(video_path: str) -> tuple[float | None, dict]:
    """Track vehicles with YOLO on densely-sampled frames so ByteTrack keeps IDs alive.
    Returns (angle_degrees | None, dict of track_id -> list of (cx, by))."""
    from ultralytics import YOLO

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    track_frames = []
    idx = 0
    while len(track_frames) < TRACK_MAX_FRAMES and idx < total:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            track_frames.append(frame)
        idx += TRACK_FRAME_STEP
    cap.release()
    print(f"[calibrate] Tracking pass: {len(track_frames)} frames (every {TRACK_FRAME_STEP} frames)")

    model = YOLO(YOLO_WEIGHTS)
    tracks: dict[int, list] = {}
    for frame in track_frames:
        res = model.track(frame, classes=[2, 3, 5, 7], persist=True, verbose=False)
        if res[0].boxes.id is None:
            continue
        ids  = res[0].boxes.id.cpu().numpy().astype(int)
        xyxy = res[0].boxes.xyxy.cpu().numpy()
        for tid, box in zip(ids, xyxy):
            cx = float((box[0] + box[2]) / 2)
            by = float(box[3])          # bottom edge ≈ wheels on road surface
            tracks.setdefault(int(tid), []).append((cx, by))

    vecs = []
    for pts in tracks.values():
        if len(pts) < 2:
            continue
        dx = pts[-1][0] - pts[0][0]
        dy = pts[-1][1] - pts[0][1]
        if math.hypot(dx, dy) > 10:    # skip nearly-stationary
            vecs.append([dx, dy])

    if not vecs:
        print("[calibrate] No moving vehicle tracks found for road-angle estimate.")
        return None, tracks

    arr = np.array(vecs, dtype=np.float32)
    _, _, vt = np.linalg.svd(arr - arr.mean(axis=0), full_matrices=False)
    dx, dy = vt[0]
    angle = math.degrees(math.atan2(abs(dy), abs(dx)))
    print(f"[calibrate] Road angle from {len(vecs)} track vectors ({len(tracks)} tracks): {angle:.1f}°")
    return float(angle), tracks


def filter_by_local_trace_angle(
    segs: np.ndarray,
    tracks: dict,
    angle_tol: float,
    search_radius: int,
) -> tuple[list, list, list]:
    """For each Hough segment find car-trace segments within search_radius.
    Keep the segment only if its angle matches those nearby traces.
    Returns (kept, rejected, orphan) — orphan = no traces nearby (kept conservatively)."""

    # flatten tracks into trace micro-segments: (midpoint_x, midpoint_y, angle_deg)
    trace_segs = []
    for pts in tracks.values():
        if len(pts) < 3:        # too short to trust — likely detection noise
            continue
        for j in range(1, len(pts)):
            x1, y1 = pts[j - 1]
            x2, y2 = pts[j]
            dx, dy  = x2 - x1, y2 - y1
            if math.hypot(dx, dy) < 5:
                continue
            trace_segs.append((
                (x1 + x2) / 2,
                (y1 + y2) / 2,
                math.degrees(math.atan2(abs(dy), abs(dx))),
            ))

    kept, rejected, orphan = [], [], []
    for seg in segs:
        x1, y1, x2, y2 = int(seg[0]), int(seg[1]), int(seg[2]), int(seg[3])
        dx, dy = x2 - x1, y2 - y1
        if dx == 0:
            rejected.append((x1, y1, x2, y2))
            continue
        seg_angle = math.degrees(math.atan2(abs(dy), abs(dx)))
        mx, my    = (x1 + x2) / 2, (y1 + y2) / 2

        if not trace_segs:
            orphan.append((x1, y1, x2, y2))
            continue

        closest = min(trace_segs, key=lambda t: math.hypot(t[0] - mx, t[1] - my))
        closest_dist = math.hypot(closest[0] - mx, closest[1] - my)

        if closest_dist > search_radius:
            orphan.append((x1, y1, x2, y2))   # nearest trace is too far → no evidence
            continue

        if abs(seg_angle - closest[2]) <= angle_tol:
            kept.append((x1, y1, x2, y2))
        else:
            rejected.append((x1, y1, x2, y2))

    return kept, rejected, orphan


def color_filter(frame: np.ndarray) -> np.ndarray:
    pixels = frame.reshape(-1, 3).astype(np.float32)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green_mask = (hsv[:, :, 0] >= 25) & (hsv[:, :, 0] <= 85) & (hsv[:, :, 1] > 40)
    non_green = pixels[~green_mask.flatten()]

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(non_green, K_COLORS, None, criteria, 3, cv2.KMEANS_RANDOM_CENTERS)

    counts = np.bincount(labels.flatten())
    dominant_cluster = int(np.argmax(counts))

    all_labels = np.full(len(pixels), -1, dtype=np.int32)
    non_green_idx = np.where(~green_mask.flatten())[0]
    all_labels[non_green_idx] = labels.flatten()

    dominant_mask = (all_labels == dominant_cluster).reshape(frame.shape[:2])

    blurred = cv2.GaussianBlur(frame, (BLUR_KERNEL, BLUR_KERNEL), 0)
    out = blurred.copy()
    out[dominant_mask] = frame[dominant_mask]
    return out


def hsv_lane_mask(frame: np.ndarray) -> np.ndarray:
    """Binary mask (255/0) of white and yellow lane marking pixels."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white  = cv2.inRange(hsv,
                         (0,            0,           WHITE_V_MIN),
                         (180,          WHITE_S_MAX, 255))
    yellow = cv2.inRange(hsv,
                         (YELLOW_H_MIN, YELLOW_S_MIN, YELLOW_V_MIN),
                         (YELLOW_H_MAX, 255,          255))
    return cv2.bitwise_or(white, yellow)


def preprocess(frame: np.ndarray, roi_top_ratio: float) -> np.ndarray:
    H, W = frame.shape[:2]
    gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred  = cv2.GaussianBlur(enhanced, (5, 5), 0)
    edges    = cv2.Canny(blurred, 50, 150)

    roi_top = int(H * roi_top_ratio)
    pts = np.array([
        [0,     H - 1],
        [W - 1, H - 1],
        [W - 1, roi_top],
        [0,     roi_top],
    ], dtype=np.int32)
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return cv2.bitwise_and(edges, edges, mask=mask)


def preprocess_directional(
    frame: np.ndarray,
    roi_top_ratio: float,
    road_angle_deg: float,
    angle_tol: float,
) -> np.ndarray:
    """Canny edges filtered to only keep pixels whose Sobel gradient direction
    is perpendicular to the road (i.e. the gradient points across lane markings)."""
    H, W = frame.shape[:2]
    gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred  = cv2.GaussianBlur(enhanced, (5, 5), 0)
    edges    = cv2.Canny(blurred, 50, 150)

    # Sobel gradient direction (0–90°, same convention as atan2(|dy|,|dx|))
    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    grad_angle = np.degrees(np.arctan2(np.abs(gy), np.abs(gx)))   # 0=horiz, 90=vert

    # Lane markings run at road_angle_deg → gradient is perpendicular: 90 − road_angle
    expected = 90.0 - road_angle_deg
    diff     = np.abs(grad_angle - expected)
    dir_mask = (diff <= angle_tol).astype(np.uint8) * 255
    edges    = cv2.bitwise_and(edges, dir_mask)

    roi_top = int(H * roi_top_ratio)
    pts = np.array([
        [0,     H - 1],
        [W - 1, H - 1],
        [W - 1, roi_top],
        [0,     roi_top],
    ], dtype=np.int32)
    roi_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(roi_mask, [pts], 255)
    return cv2.bitwise_and(edges, edges, mask=roi_mask)


def detect_lines(
    edge_img: np.ndarray,
    threshold: int,
    min_length: int,
    max_gap: int,
) -> np.ndarray | None:
    lines = cv2.HoughLinesP(
        edge_img, 1, np.pi / 180,
        threshold=threshold,
        minLineLength=min_length,
        maxLineGap=max_gap,
    )
    if lines is None:
        return None
    return lines.reshape(-1, 4)


def group_road_markers(
    lines: np.ndarray,
    H: int, W: int,
    n_groups: int,
    angle_min_deg: float,
    angle_max_deg: float,
) -> list[list]:
    """Cluster segments by direction + x-position at frame bottom.
    Returns groups sorted left-to-right (index 0 = leftmost lane line)."""
    valid = []
    for x1, y1, x2, y2 in lines:
        dx, dy = x2 - x1, y2 - y1
        if dx == 0:
            continue
        angle = math.degrees(math.atan2(abs(dy), abs(dx)))
        if not (angle_min_deg <= angle <= angle_max_deg):
            continue
        valid.append((int(x1), int(y1), int(x2), int(y2)))

    if len(valid) < n_groups:
        return []

    def x_at_bottom(x1, y1, x2, y2):
        dx, dy = x2 - x1, y2 - y1
        slope = dy / dx
        b = y1 - slope * x1
        return (H - 1 - b) / slope if abs(slope) > 1e-6 else (x1 + x2) / 2

    features = np.array(
        [[math.degrees(math.atan2(abs(y2-y1), abs(x2-x1))) / 90.0,
          x_at_bottom(x1, y1, x2, y2) / W]
         for x1, y1, x2, y2 in valid],
        dtype=np.float32,
    )

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    _, labels, _ = cv2.kmeans(features, n_groups, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)
    labels = labels.flatten()

    raw_groups = {}
    for i, seg in enumerate(valid):
        raw_groups.setdefault(labels[i], []).append(seg)

    def mean_x_bot(segs):
        return sum(x_at_bottom(*s) for s in segs) / len(segs)

    return sorted(raw_groups.values(), key=mean_x_bot)


def fit_line(segs: list) -> tuple[float, float]:
    x_pts, y_pts = [], []
    for x1, y1, x2, y2 in segs:
        x_pts.extend([x1, x2])
        y_pts.extend([y1, y2])
    m, b = np.polyfit(x_pts, y_pts, 1)
    return float(m), float(b)


def compute_vanishing_point(
    left_mb: tuple[float, float],
    right_mb: tuple[float, float],
) -> tuple[float, float] | None:
    m1, b1 = left_mb
    m2, b2 = right_mb
    denom = m1 - m2
    if abs(denom) < 1e-6:
        return None
    vp_x = (b2 - b1) / denom
    vp_y = m1 * vp_x + b1
    return float(vp_x), float(vp_y)


def validate_vp(vp_x: float, vp_y: float, H: int, W: int) -> bool:
    return (vp_y < H * 0.5) and (W * 0.1 < vp_x < W * 0.9)


def fallback_vp(
    left_mb: tuple[float, float],
    right_mb: tuple[float, float],
    y_top: int,
) -> tuple[float, float]:
    m_l, b_l = left_mb
    m_r, b_r = right_mb
    safe_y = y_top * 0.5
    xl = (safe_y - b_l) / m_l
    xr = (safe_y - b_r) / m_r
    return (xl + xr) / 2.0, float(safe_y)


def build_grid(
    frame_shape,
    vp_x: float, vp_y: float,
    left_mb: tuple, right_mb: tuple,
    y_bottom: int, y_top: int,
    num_cols: int, num_rows: int,
) -> dict:
    m_l, b_l = left_mb
    m_r, b_r = right_mb
    left_x_bot  = (y_bottom - b_l) / m_l
    right_x_bot = (y_bottom - b_r) / m_r

    vertical_lines = []
    for i in range(num_cols + 1):
        x_bot = left_x_bot + i * (right_x_bot - left_x_bot) / num_cols
        vertical_lines.append([float(x_bot), float(y_bottom), float(vp_x), float(vp_y)])

    if vp_y >= y_top:
        vp_y = y_top * 0.5
    R = (y_bottom - vp_y) / (y_top - vp_y)
    horiz_ys = []
    for k in range(num_rows + 1):
        y_k = vp_y + (y_bottom - vp_y) / (1.0 + k * (R - 1.0) / num_rows)
        horiz_ys.append(float(y_k))

    H, W = frame_shape[:2]
    return {
        "camera_id":            CAMERA_ID,
        "frame_shape":          [int(H), int(W)],
        "vanishing_point":      [float(vp_x), float(vp_y)],
        "left_line":            {"m": float(m_l), "b": float(b_l)},
        "right_line":           {"m": float(m_r), "b": float(b_r)},
        "grid_top_y":           int(y_top),
        "grid_bottom_y":        int(y_bottom),
        "num_cols":             int(num_cols),
        "num_rows":             int(num_rows),
        "vertical_lines":       vertical_lines,
        "horizontal_y_values":  horiz_ys,
    }


def show_preview(frame: np.ndarray, config: dict) -> None:
    preview = frame.copy()
    draw_grid(preview, config, alpha=0.5, color=(0, 255, 0))
    cv2.imshow("Grid preview — any key to accept, q to quit", preview)
    key = cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()
    if key == ord("q"):
        sys.exit("[calibrate] Calibration cancelled by user.")


def save_config(config: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"[calibrate] Saved grid config to {path}")


def dedup_segs(segs: list, dist_thr: float, angle_thr: float) -> list:
    """Drop segments whose midpoint and angle are too close to an already-kept segment."""
    result = []
    for seg in segs:
        x1, y1, x2, y2 = seg
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        dx, dy = x2 - x1, y2 - y1
        ang = math.degrees(math.atan2(abs(dy), abs(dx))) if abs(dx) > 0 else 90.0
        dup = any(
            math.hypot(mx - (rx1 + rx2) / 2, my - (ry1 + ry2) / 2) < dist_thr
            and abs(ang - (math.degrees(math.atan2(abs(ry2 - ry1), abs(rx2 - rx1)))
                           if abs(rx2 - rx1) > 0 else 90.0)) < angle_thr
            for rx1, ry1, rx2, ry2 in result
        )
        if not dup:
            result.append(seg)
    return result


def _dbg(name: str, img) -> None:
    if DEBUG:
        path = f"debug_{name}.png"
        cv2.imwrite(path, img)
        print(f"[debug] saved {path}")


def main():
    frames = sample_frames(VIDEO_PATH)
    first_frame = frames[0]
    H, W = first_frame.shape[:2]
    y_bottom = H - 1
    y_top    = int(H * GRID_TOP_RATIO)

    road_angle, tracks = compute_road_angle(VIDEO_PATH)

    if DEBUG and tracks:
        dbg_trace = first_frame.copy()
        for t_idx, (tid, pts) in enumerate(tracks.items()):
            hue = int((t_idx * 137) % 180)
            hsv_px = np.uint8([[[hue, 220, 220]]])
            color = cv2.cvtColor(hsv_px, cv2.COLOR_HSV2BGR)[0, 0].tolist()
            for j in range(1, len(pts)):
                p1 = (round(pts[j - 1][0]), round(pts[j - 1][1]))
                p2 = (round(pts[j][0]),     round(pts[j][1]))
                cv2.line(dbg_trace, p1, p2, color, 1, cv2.LINE_AA)
                dx, dy = pts[j][0] - pts[j-1][0], pts[j][1] - pts[j-1][1]
                if math.hypot(dx, dy) > 5:
                    seg_ang = math.degrees(math.atan2(abs(dy), abs(dx)))
                    mx = round((pts[j-1][0] + pts[j][0]) / 2)
                    my = round((pts[j-1][1] + pts[j][1]) / 2)
                    cv2.putText(dbg_trace, f"{seg_ang:.0f}",
                                (mx, my), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.LINE_AA)
            cv2.circle(dbg_trace, (round(pts[-1][0]), round(pts[-1][1])), 4, color, -1, cv2.LINE_AA)
            cv2.putText(dbg_trace, str(tid),
                        (round(pts[-1][0]) + 5, round(pts[-1][1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
        if road_angle is not None:
            vecs = []
            for pts in tracks.values():
                if len(pts) >= 2:
                    vecs.append([pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1]])
            if vecs:
                arr_v = np.array(vecs, dtype=np.float32)
                _, _, vt = np.linalg.svd(arr_v - arr_v.mean(axis=0), full_matrices=False)
                dv = vt[0] / (np.linalg.norm(vt[0]) + 1e-9)
                mx = sum(p[0] for pts in tracks.values() for p in pts) / max(1, sum(len(p) for p in tracks.values()))
                my = sum(p[1] for pts in tracks.values() for p in pts) / max(1, sum(len(p) for p in tracks.values()))
                scale = min(H, W) // 4
                p1 = (round(mx - dv[0] * scale), round(my - dv[1] * scale))
                p2 = (round(mx + dv[0] * scale), round(my + dv[1] * scale))
                #cv2.arrowedLine(dbg_trace, p1, p2, (255, 255, 255), 3, cv2.LINE_AA, tipLength=0.2)
                cv2.putText(dbg_trace, f"road {road_angle:.1f}deg",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        _dbg("0_car_traces", dbg_trace)

    # accumulate segments — use directional Canny when road_angle is known
    all_segs = []
    for frame in frames:
        filtered = color_filter(frame)
        if road_angle is not None:
            edges = preprocess_directional(filtered, ROI_TOP_RATIO, road_angle, ROAD_ANGLE_TOL)
        else:
            edges = preprocess(filtered, ROI_TOP_RATIO)
        edges = cv2.bitwise_and(edges, hsv_lane_mask(frame))   # keep only white/yellow edges
        lines = detect_lines(edges, HOUGH_THRESHOLD, MIN_LINE_LENGTH, MAX_LINE_GAP)
        if lines is not None:
            all_segs.extend(map(tuple, lines))

    print(f"[calibrate] Accumulated {len(all_segs)} segments from {len(frames)} frames"
          + (" (directional Canny)" if road_angle is not None else ""))

    if len(all_segs) < N_LANE_LINES:
        sys.exit("[calibrate] ERROR: Not enough line segments found. Lower HOUGH_THRESHOLD.")

    all_segs_arr = np.array(all_segs, dtype=np.int32)

    kept, rejected, orphan = filter_by_local_trace_angle(
        all_segs_arr, tracks, ROAD_ANGLE_TOL, SEARCH_RADIUS
    )
    print(f"[calibrate] Local trace filter: {len(kept)} kept, {len(rejected)} rejected, "
          f"{len(orphan)} orphan (no nearby trace)")

    if DEBUG:
        dbg_filter = first_frame.copy()
        def _draw_segs_labeled(img, segs, color, lw):
            for x1, y1, x2, y2 in segs:
                cv2.line(img, (x1, y1), (x2, y2), color, lw, cv2.LINE_AA)
                dx, dy = x2 - x1, y2 - y1
                if abs(dx) > 0:
                    ang = math.degrees(math.atan2(abs(dy), abs(dx)))
                    cv2.putText(img, f"{ang:.0f}",
                                ((x1 + x2) // 2, (y1 + y2) // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.LINE_AA)
        _draw_segs_labeled(dbg_filter, rejected, (0, 0, 220),   1)
        _draw_segs_labeled(dbg_filter, orphan,   (0, 165, 255), 1)
        _draw_segs_labeled(dbg_filter, kept,     (0, 220, 0),   1)
        cv2.putText(dbg_filter, "green=kept  orange=orphan  red=rejected",
                    (10, H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        _dbg("3_angle_filter", dbg_filter)

    pass_segs = kept + orphan
    if len(pass_segs) < N_LANE_LINES:
        sys.exit("[calibrate] ERROR: Not enough segments after local trace filter.")
    pass_arr = np.array(pass_segs, dtype=np.int32)
    groups = group_road_markers(pass_arr, H, W, N_LANE_LINES, ANGLE_MIN_DEG, ANGLE_MAX_DEG)

    if len(groups) < 2:
        sys.exit("[calibrate] ERROR: Could not find 2 distinct lane marker groups.")

    # leftmost group = left lane, rightmost = right lane
    left_segs  = groups[0]
    right_segs = groups[-1]
    print(f"[calibrate] Groups: {[len(g) for g in groups]} segments each (left→right)")

    # debug images using first frame
    if DEBUG:
        _dbg("1_color_filter", color_filter(first_frame))
        _edge_img = (preprocess_directional(color_filter(first_frame), ROI_TOP_RATIO, road_angle, ROAD_ANGLE_TOL)
                     if road_angle is not None else
                     preprocess(color_filter(first_frame), ROI_TOP_RATIO))
        _edge_img = cv2.bitwise_and(_edge_img, hsv_lane_mask(first_frame))
        _dbg("2_edges", _edge_img)
        # distinct color per group using golden-ratio hue spread
        dbg_lines = first_frame.copy()
        palette = []
        for i in range(len(groups)):
            hue = int((i * 137) % 180)
            hsv_px = np.uint8([[[hue, 220, 220]]])
            palette.append(cv2.cvtColor(hsv_px, cv2.COLOR_HSV2BGR)[0, 0].tolist())
        kept_set = set(kept)
        for g_idx, segs in enumerate(groups):
            col = palette[g_idx]
            kept_in_group = [s for s in segs if tuple(s) in kept_set]
            for x1, y1, x2, y2 in dedup_segs(kept_in_group, DEDUP_DIST, DEDUP_ANGLE):
                cv2.line(dbg_lines, (x1, y1), (x2, y2), col, 1, cv2.LINE_AA)
                dx, dy = x2 - x1, y2 - y1
                if abs(dx) > 0:
                    ang = math.degrees(math.atan2(abs(dy), abs(dx)))
                    cv2.putText(dbg_lines, f"{ang:.0f}",
                                ((x1 + x2) // 2, (y1 + y2) // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, col, 1, cv2.LINE_AA)
            # draw fitted line across frame height using only kept segments
            if len(kept_in_group) >= 2:
                m, b = fit_line(kept_in_group)
                y1d, y2d = H - 1, int(H * GRID_TOP_RATIO)
                x1d = int((y1d - b) / m) if abs(m) > 1e-6 else 0
                x2d = int((y2d - b) / m) if abs(m) > 1e-6 else 0
                cv2.line(dbg_lines, (x1d, y1d), (x2d, y2d), col, 2, cv2.LINE_AA)
        _dbg("3_lines", dbg_lines)

    left_mb  = fit_line(left_segs)
    right_mb = fit_line(right_segs)

    result = compute_vanishing_point(left_mb, right_mb)
    if result is None or not validate_vp(result[0], result[1], H, W):
        print("[calibrate] WARNING: Vanishing point outside valid region. Using fallback.")
        vp_x, vp_y = fallback_vp(left_mb, right_mb, y_top)
    else:
        vp_x, vp_y = result

    config = build_grid(first_frame.shape, vp_x, vp_y, left_mb, right_mb, y_bottom, y_top, NUM_COLS, NUM_ROWS)

    if DEBUG:
        dbg_grid = first_frame.copy()
        draw_grid(dbg_grid, config, alpha=0.6, color=(0, 255, 0))
        _dbg("4_grid", dbg_grid)

    show_preview(first_frame, config)

    out_path = f"grid_config_{CAMERA_ID}.json"
    save_config(config, out_path)


if __name__ == "__main__":
    main()
