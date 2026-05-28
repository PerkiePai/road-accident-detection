"""Measure real-world lane width by back-projecting two clicked pixels.

Assumes focal_length.py has been run first, producing K_unidepth.npy,
depth_unidepth.npy, and frame_unidepth.png.

UniDepth's metric scale drifts on out-of-distribution cameras; this tool
also lets you calibrate that scale against a known real-world distance
and persists the factor in scale_unidepth.npy.

Controls:
    left-click   : add a point (max 2)
    r            : reset points
    Enter / space: compute width with current 2 points
    c            : calibrate scale from last measurement (asks expected dist)
    q / Esc      : quit
"""

import sys
from pathlib import Path

import cv2
import numpy as np

K_PATH = Path("K_unidepth.npy")
DEPTH_PATH = Path("depth_unidepth.npy")
FRAME_PATH = Path("frame_unidepth.png")
SCALE_PATH = Path("scale_unidepth.npy")


def backproject(u: int, v: int, depth_map: np.ndarray, K: np.ndarray) -> np.ndarray:
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    Z = float(depth_map[v, u])
    X = (u - cx) / fx * Z
    Y = (v - cy) / fy * Z
    return np.array([X, Y, Z], dtype=np.float64)


def load_scale() -> float:
    if SCALE_PATH.exists():
        return float(np.load(SCALE_PATH))
    return 1.0


def save_scale(s: float) -> None:
    np.save(SCALE_PATH, np.float32(s))


def main() -> None:
    for p in (K_PATH, DEPTH_PATH, FRAME_PATH):
        if not p.exists():
            sys.exit(f"missing {p} — run focal_length.py first")

    K = np.load(K_PATH)
    depth_raw = np.load(DEPTH_PATH)
    scale = load_scale()
    print(f"loaded scale = {scale:.4f}  (1.0 = no correction)")
    depth = depth_raw * scale
    frame = cv2.imread(str(FRAME_PATH))
    H, W = frame.shape[:2]
    assert depth.shape == (H, W), f"depth {depth.shape} != frame {(H, W)}"

    points: list[tuple[int, int]] = []
    last_measurement: float | None = None

    def redraw() -> np.ndarray:
        canvas = frame.copy()
        for i, (u, v) in enumerate(points):
            cv2.circle(canvas, (u, v), 6, (0, 255, 0), -1)
            cv2.putText(canvas, f"{i+1}: Z={depth[v, u]:.2f}m",
                        (u + 10, v - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 1, cv2.LINE_AA)
        if len(points) == 2:
            cv2.line(canvas, points[0], points[1], (0, 255, 255), 2)
        return canvas

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((x, y))
            cv2.imshow("lane_width", redraw())

    cv2.namedWindow("lane_width", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("lane_width", on_mouse)
    cv2.imshow("lane_width", redraw())

    print("click left edge then right edge of a lane.")
    print("Enter = measure, c = calibrate from last measure, r = reset, q = quit.")

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("r"):
            points.clear()
            cv2.imshow("lane_width", redraw())
        if key in (13, 32) and len(points) == 2:
            p1 = backproject(*points[0], depth, K)
            p2 = backproject(*points[1], depth, K)
            dist = float(np.linalg.norm(p1 - p2))
            dx, dy, dz = p2 - p1
            last_measurement = dist
            print()
            print(f"pixel 1: {points[0]}  ->  X={p1[0]:+.2f}  Y={p1[1]:+.2f}  Z={p1[2]:.2f} m")
            print(f"pixel 2: {points[1]}  ->  X={p2[0]:+.2f}  Y={p2[1]:+.2f}  Z={p2[2]:.2f} m")
            print(f"delta  : dX={dx:+.2f}  dY={dy:+.2f}  dZ={dz:+.2f} m")
            print(f"3D distance: {dist:.2f} m  (current scale = {scale:.4f})")
        if key == ord("c"):
            if last_measurement is None:
                print("nothing to calibrate against — measure something first.")
                continue
            try:
                raw = input(f"expected real-world distance in meters "
                            f"(last measured {last_measurement:.2f}): ").strip()
                expected = float(raw)
            except ValueError:
                print("invalid number, calibration aborted.")
                continue
            if expected <= 0 or last_measurement <= 0:
                print("non-positive value, calibration aborted.")
                continue
            new_scale = scale * expected / last_measurement
            save_scale(new_scale)
            print(f"calibrated: scale {scale:.4f} -> {new_scale:.4f}  "
                  f"(saved to {SCALE_PATH})")
            print("re-run the script to apply.")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
