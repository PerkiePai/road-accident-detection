import sys
import json
import math
import cv2
import numpy as np

from grid_overlay import draw_grid

VIDEO_PATH      = "in/thai_road_full.mp4"
CAMERA_ID       = "cam_01"
NUM_COLS        = 3
NUM_ROWS        = 8
ROI_TOP_RATIO   = 0.35
GRID_TOP_RATIO  = 0.40
HOUGH_THRESHOLD = 50
MIN_LINE_LENGTH = 50
MAX_LINE_GAP    = 20
ANGLE_MIN_DEG   = 15
ANGLE_MAX_DEG   = 75


def grab_frame(video_path: str) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        sys.exit(f"[calibrate] ERROR: Could not read frame from {video_path}")
    return frame


def preprocess(frame: np.ndarray, roi_top_ratio: float) -> np.ndarray:
    H, W = frame.shape[:2]
    gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred  = cv2.GaussianBlur(enhanced, (5, 5), 0)
    edges    = cv2.Canny(blurred, 50, 150)

    roi_top = int(H * roi_top_ratio)
    pts = np.array([
        [0,          H - 1],
        [W - 1,      H - 1],
        [int(W * 0.65), roi_top],
        [int(W * 0.35), roi_top],
    ], dtype=np.int32)
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return cv2.bitwise_and(edges, edges, mask=mask)


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


def filter_and_group(
    lines: np.ndarray,
    angle_min_deg: float,
    angle_max_deg: float,
) -> tuple[list, list]:
    left_segs, right_segs = [], []
    for x1, y1, x2, y2 in lines:
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0:
            continue
        angle = math.degrees(math.atan2(abs(dy), abs(dx)))
        if not (angle_min_deg <= angle <= angle_max_deg):
            continue
        slope = dy / dx
        if slope < 0:
            left_segs.append((x1, y1, x2, y2))
        elif slope > 0:
            right_segs.append((x1, y1, x2, y2))
    return left_segs, right_segs


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
    xl = (y_top - b_l) / m_l
    xr = (y_top - b_r) / m_r
    return (xl + xr) / 2.0, float(y_top)


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


def main():
    frame = grab_frame(VIDEO_PATH)
    H, W = frame.shape[:2]
    y_bottom = H - 1
    y_top    = int(H * GRID_TOP_RATIO)

    edges = preprocess(frame, ROI_TOP_RATIO)
    lines = detect_lines(edges, HOUGH_THRESHOLD, MIN_LINE_LENGTH, MAX_LINE_GAP)

    if lines is None:
        sys.exit(
            "[calibrate] ERROR: HoughLinesP found no lines.\n"
            "  Try: lower HOUGH_THRESHOLD, increase MAX_LINE_GAP, or select a different frame."
        )

    left_segs, right_segs = filter_and_group(lines, ANGLE_MIN_DEG, ANGLE_MAX_DEG)

    if not left_segs or not right_segs:
        missing = "left" if not left_segs else "right"
        sys.exit(
            f"[calibrate] ERROR: No {missing} lane lines found after angle filtering.\n"
            f"  Adjust ANGLE_MIN_DEG/ANGLE_MAX_DEG or HOUGH_THRESHOLD."
        )

    left_mb  = fit_line(left_segs)
    right_mb = fit_line(right_segs)

    result = compute_vanishing_point(left_mb, right_mb)
    if result is None or not validate_vp(result[0], result[1], H, W):
        print("[calibrate] WARNING: Vanishing point outside valid region. Using fallback.")
        vp_x, vp_y = fallback_vp(left_mb, right_mb, y_top)
    else:
        vp_x, vp_y = result

    config = build_grid(frame.shape, vp_x, vp_y, left_mb, right_mb, y_bottom, y_top, NUM_COLS, NUM_ROWS)

    show_preview(frame, config)

    out_path = f"grid_config_{CAMERA_ID}.json"
    save_config(config, out_path)


if __name__ == "__main__":
    main()
