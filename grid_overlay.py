import json
import cv2
import numpy as np

GRID_ALPHA  = 0.4
GRID_COLOR  = (0, 255, 0)
LABEL_CELLS = False


def load_grid(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return json.load(f)


def _vert_x_at_y(x_bot: float, y_bot: float, vp_x: float, vp_y: float, y_k: float) -> float:
    t = (y_k - y_bot) / (vp_y - y_bot)
    return x_bot + t * (vp_x - x_bot)


def draw_grid(
    frame: np.ndarray,
    config: dict,
    alpha: float = GRID_ALPHA,
    color: tuple = GRID_COLOR,
) -> None:
    vp_x, vp_y       = config["vanishing_point"]
    y_bottom          = config["grid_bottom_y"]
    y_top             = config["grid_top_y"]
    vertical_lines    = config["vertical_lines"]
    horiz_ys          = config["horizontal_y_values"]
    num_cols          = config["num_cols"]
    num_rows          = config["num_rows"]

    overlay = frame.copy()

    for vl in vertical_lines:
        x_bot, y_bot = vl[0], vl[1]
        x_top = _vert_x_at_y(x_bot, y_bot, vp_x, vp_y, y_top)
        cv2.line(
            overlay,
            (round(x_bot), round(y_bot)),
            (round(x_top), int(y_top)),
            color, 1, cv2.LINE_AA,
        )

    for y_k in horiz_ys:
        xl = _vert_x_at_y(vertical_lines[0][0],  y_bottom, vp_x, vp_y, y_k)
        xr = _vert_x_at_y(vertical_lines[-1][0], y_bottom, vp_x, vp_y, y_k)
        cv2.line(
            overlay,
            (round(xl), round(y_k)),
            (round(xr), round(y_k)),
            color, 1, cv2.LINE_AA,
        )

    if LABEL_CELLS:
        for row in range(num_rows):
            for col in range(num_cols):
                y_mid = (horiz_ys[row] + horiz_ys[row + 1]) / 2
                xl = _vert_x_at_y(vertical_lines[col][0],     y_bottom, vp_x, vp_y, y_mid)
                xr = _vert_x_at_y(vertical_lines[col + 1][0], y_bottom, vp_x, vp_y, y_mid)
                x_mid = (xl + xr) / 2
                cv2.putText(
                    overlay,
                    f"r{row}c{col}",
                    (round(x_mid) - 15, round(y_mid) + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA,
                )

    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
