"""Predict camera intrinsics from a single video frame using UniDepth.

UniDepth (Piccinelli et al., CVPR 2024) jointly predicts metric depth and
camera intrinsics from a single RGB image — no scale reference needed.

Install:
    pip install git+https://github.com/lpiccinelli-eth/UniDepth.git

Outputs:
    K_unidepth.npy      -- 3x3 intrinsic matrix
    depth_unidepth.npy  -- HxW float32 metric depth (meters)
    frame_unidepth.png  -- the exact RGB frame used (for downstream tools)
    debug_depth.png     -- depth map visualization side-by-side with RGB
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

VIDEO = Path("in/car_100kmh.mp4")
FRAME_INDEX = 60  # ~2s into the clip
MODEL_NAME = "lpiccinelli/unidepth-v2-vitl14"

OUT_K = Path("K_unidepth.npy")
OUT_DEPTH = Path("depth_unidepth.npy")
OUT_FRAME = Path("frame_unidepth.png")
OUT_DEPTH_VIS = Path("debug_depth.png")


def grab_frame(video_path: Path, idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit(f"could not open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        sys.exit(f"could not read frame {idx}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def colorize_depth(depth: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(depth, [2, 98])
    norm = np.clip((depth - lo) / (hi - lo + 1e-9), 0, 1)
    vis = (norm * 255).astype(np.uint8)
    return cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO)


def main() -> None:
    try:
        from unidepth.models import UniDepthV2
    except ImportError:
        sys.exit(
            "unidepth not installed. run:\n"
            "  pip install git+https://github.com/lpiccinelli-eth/UniDepth.git"
        )

    if not VIDEO.exists():
        sys.exit(f"video not found: {VIDEO}")

    rgb = grab_frame(VIDEO, FRAME_INDEX)
    H, W = rgb.shape[:2]
    print(f"frame      : {VIDEO} #{FRAME_INDEX}  ({W}x{H})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device     : {device}")

    model = UniDepthV2.from_pretrained(MODEL_NAME).to(device).eval()

    rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).to(device)  # (3,H,W) uint8

    with torch.inference_mode():
        pred = model.infer(rgb_tensor)

    K = pred["intrinsics"].squeeze(0).cpu().numpy()       # (3,3)
    depth = pred["depth"].squeeze().cpu().numpy()          # (H,W)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    print()
    print("predicted intrinsics (pixels):")
    print(f"  fx = {fx:.2f}")
    print(f"  fy = {fy:.2f}")
    print(f"  cx = {cx:.2f}   (image center: {W/2:.1f})")
    print(f"  cy = {cy:.2f}   (image center: {H/2:.1f})")
    hfov = np.degrees(2 * np.arctan(W / (2 * fx)))
    vfov = np.degrees(2 * np.arctan(H / (2 * fy)))
    print(f"  horizontal FOV ~ {hfov:.1f} deg")
    print(f"  vertical FOV   ~ {vfov:.1f} deg")

    print()
    print(f"depth stats: min={depth.min():.2f} m, "
          f"max={depth.max():.2f} m, median={np.median(depth):.2f} m")

    np.save(OUT_K, K)
    np.save(OUT_DEPTH, depth.astype(np.float32))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(OUT_FRAME), bgr)
    side = np.hstack([bgr, colorize_depth(depth)])
    cv2.imwrite(str(OUT_DEPTH_VIS), side)
    print(f"saved      : {OUT_K}")
    print(f"saved      : {OUT_DEPTH}")
    print(f"saved      : {OUT_FRAME}")
    print(f"saved      : {OUT_DEPTH_VIS}")


if __name__ == "__main__":
    main()
