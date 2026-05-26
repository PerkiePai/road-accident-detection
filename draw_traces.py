import cv2
import json
import numpy as np

VIDEO_PATH = "in/car_long.mp4"
MOCK_PATH  = "mock_tracks.json"

cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()
cap.release()
if not ret:
    raise RuntimeError(f"Cannot read {VIDEO_PATH}")

tracks: dict[int, list] = {}
current_tid  = 0
current_pts: list = []


def _color(tid: int) -> tuple:
    hue = int((tid * 137) % 180)
    px  = np.uint8([[[hue, 220, 220]]])
    return cv2.cvtColor(px, cv2.COLOR_HSV2BGR)[0, 0].tolist()


def redraw() -> np.ndarray:
    img = frame.copy()
    for tid, pts in tracks.items():
        col = _color(tid)
        for j in range(1, len(pts)):
            cv2.line(img, tuple(pts[j-1]), tuple(pts[j]), col, 2, cv2.LINE_AA)
        if pts:
            cv2.circle(img, tuple(pts[-1]), 5, col, -1, cv2.LINE_AA)
            cv2.putText(img, str(tid), (pts[-1][0]+6, pts[-1][1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
    # current in-progress trace
    for j in range(1, len(current_pts)):
        cv2.line(img, tuple(current_pts[j-1]), tuple(current_pts[j]),
                 (0, 255, 255), 2, cv2.LINE_AA)
    if current_pts:
        cv2.circle(img, tuple(current_pts[-1]), 5, (0, 255, 255), -1, cv2.LINE_AA)
    cv2.putText(img,
                f"trace {current_tid}  pts={len(current_pts)} | "
                "LClick=add  N=new trace  U=undo  S=save  Q=quit",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        current_pts.append([x, y])
        cv2.imshow("Draw Traces", redraw())


cv2.namedWindow("Draw Traces")
cv2.setMouseCallback("Draw Traces", on_mouse)
cv2.imshow("Draw Traces", redraw())

while True:
    key = cv2.waitKey(0) & 0xFF
    if key == ord('n'):                          # finish current trace, start new
        if len(current_pts) >= 2:
            tracks[current_tid] = [list(p) for p in current_pts]
            current_tid += 1
        current_pts.clear()
        cv2.imshow("Draw Traces", redraw())
    elif key == ord('u'):                        # undo last point
        if current_pts:
            current_pts.pop()
            cv2.imshow("Draw Traces", redraw())
    elif key == ord('s'):                        # save
        if len(current_pts) >= 2:
            tracks[current_tid] = [list(p) for p in current_pts]
        with open(MOCK_PATH, "w") as f:
            json.dump({str(k): v for k, v in tracks.items()}, f, indent=2)
        print(f"Saved {len(tracks)} trace(s) to {MOCK_PATH}")
        cv2.imshow("Draw Traces", redraw())
    elif key == ord('q'):
        break

cv2.destroyAllWindows()
