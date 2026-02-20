import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_multiotsu

# Load image using OpenCV (BGR)
img = cv2.imread("images/scratch_2.tiff")

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur
blur = cv2.medianBlur(gray, 27)

# Number of classes (regions)
# e.g. 3 classes -> 2 thresholds
classes = 4

# Compute multi-level Otsu thresholds
thresholds = threshold_multiotsu(blur, classes=classes)

print("Optimal thresholds:", thresholds)

# Digitize image into regions
segmented = np.digitize(blur, bins=thresholds)

# Map regions to intensities for visualization
values = np.linspace(0, 255, classes).astype(np.uint8)
segmented_img = values[segmented]

# Apply Laplacian edge detection
laplacian = cv2.Laplacian(segmented_img, cv2.CV_64F)
edges = cv2.convertScaleAbs(laplacian)

# Detect points that form a line
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 68, minLineLength=300, maxLineGap=50)

for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)


# Display results
fig, ax = plt.subplots(1, 5, figsize=(12, 4))

ax[0].imshow(blur, cmap="gray")
ax[0].set_title("Grayscale")

ax[1].imshow(segmented, cmap="gray")
ax[1].set_title("Segmented (labels)")

ax[2].imshow(segmented_img, cmap="gray")
ax[2].set_title("Segmented (visualized)")

ax[3].imshow(edges, cmap="gray")
ax[3].set_title("Laplacian convolution")

ax[4].imshow(img, cmap="gray")
ax[4].set_title("Hough Lines")

for a in ax:
    a.axis("off")

plt.tight_layout()
plt.show()
