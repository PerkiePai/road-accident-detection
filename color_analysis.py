import cv2
import numpy as np

VIDEO_PATH = "in/accident_cm_in_p5.mp4"
K = 2

cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()
cap.release()

pixels = frame.reshape(-1, 3).astype(np.float32)

# remove green pixels (roadside trees) before clustering
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
green_mask = (hsv[:, :, 0] >= 25) & (hsv[:, :, 0] <= 85) & (hsv[:, :, 1] > 40)
non_green = pixels[~green_mask.flatten()]
print(f"Kept {len(non_green):,} / {len(pixels):,} pixels ({100*len(non_green)/len(pixels):.1f}% non-green)\n")

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
_, labels, centers = cv2.kmeans(non_green, K, None, criteria, 3, cv2.KMEANS_RANDOM_CENTERS)

counts = np.bincount(labels.flatten())
order = np.argsort(-counts)

print(f"{'Rank':<5} {'%':>6}  {'B':>4} {'G':>4} {'R':>4}  Hex")
print("-" * 40)
for rank, i in enumerate(order):
    b, g, r = centers[i].astype(int)
    pct = 100 * counts[i] / len(labels)
    hex_color = f"#{r:02X}{g:02X}{b:02X}"
    print(f"{rank+1:<5} {pct:>5.1f}%  {b:>4} {g:>4} {r:>4}  {hex_color}")

# save swatch image so you can see the colors visually
swatch_h, swatch_w = 80, 960
swatch = np.zeros((swatch_h, swatch_w, 3), dtype=np.uint8)
x = 0
for i in order:
    w = int(swatch_w * counts[i] / len(labels))
    swatch[:, x:x+w] = centers[i].astype(np.uint8)
    x += w
cv2.imwrite("color_swatch.png", swatch)
print("\nSaved color_swatch.png")

# --- output PNG showing only dominant-color pixels, black elsewhere ---
# map cluster labels back to all pixels (-1 = green/excluded)
all_labels = np.full(len(pixels), -1, dtype=np.int32)
non_green_idx = np.where(~green_mask.flatten())[0]
all_labels[non_green_idx] = labels.flatten()

dominant_cluster = order[0]
dominant_mask = (all_labels == dominant_cluster).reshape(frame.shape[:2])

blurred = cv2.GaussianBlur(frame, (51, 51), 0)
out = blurred.copy()
out[dominant_mask] = frame[dominant_mask]
cv2.imwrite("dominant_color.png", out)
print("Saved dominant_color.png  (dominant cluster = sharp, rest = blurred)")
