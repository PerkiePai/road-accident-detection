import math
import cv2
import numpy as np
from calibrate import color_filter, VIDEO_PATH, ROI_TOP_RATIO

cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()
cap.release()
H, W = frame.shape[:2]
print(f"Frame: {W}x{H}")

filtered = color_filter(frame)
cv2.imwrite("debug_color_filter.png", filtered)
print("Saved debug_color_filter.png")

gray     = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced = clahe.apply(gray)
blurred  = cv2.GaussianBlur(enhanced, (5, 5), 0)
edges    = cv2.Canny(blurred, 50, 150)

roi_top = int(H * ROI_TOP_RATIO)
pts = np.array([
    [0,          H - 1],
    [W - 1,      H - 1],
    [int(W * 0.65), roi_top],
    [int(W * 0.35), roi_top],
], dtype=np.int32)
mask = np.zeros((H, W), dtype=np.uint8)
cv2.fillPoly(mask, [pts], 255)
masked_edges = cv2.bitwise_and(edges, edges, mask=mask)

cv2.imwrite("debug_edges.png", masked_edges)
print("Saved debug_edges.png")

lines = cv2.HoughLinesP(
    masked_edges, 1, np.pi / 180,
    threshold=50, minLineLength=50, maxLineGap=20,
)

if lines is None:
    print("NO LINES FOUND AT ALL")
else:
    lines = lines.reshape(-1, 4)
    print(f"Total Hough lines: {len(lines)}")
    debug = frame.copy()
    for x1, y1, x2, y2 in lines:
        dx, dy = x2 - x1, y2 - y1
        if dx == 0:
            continue
        angle = math.degrees(math.atan2(abs(dy), abs(dx)))
        slope = dy / dx
        if slope < 0 and 15 <= angle <= 75:
            color = (0, 0, 255)       # red  = left lane (kept)
        elif slope > 0 and 15 <= angle <= 75:
            color = (0, 255, 0)       # green = right lane (kept)
        else:
            color = (128, 128, 128)   # gray  = filtered out
        cv2.line(debug, (x1, y1), (x2, y2), color, 2)
        print(f"slope={slope:+.3f}  angle={angle:.1f}deg  ({x1},{y1})->({x2},{y2})")
    cv2.imwrite("debug_lines.png", debug)
    print("Saved debug_lines.png  (red=left kept, green=right kept, gray=filtered out)")
