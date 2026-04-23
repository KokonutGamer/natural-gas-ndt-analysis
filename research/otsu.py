import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_multiotsu

# Load image using OpenCV (BGR)
img = cv2.imread("images/scratch_2.tiff")

assert img is not None, "Image could not be loaded. Check the file path."

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Number of classes (regions)
# e.g. 3 classes -> 2 thresholds
classes = 3

# Compute multi-level Otsu thresholds
thresholds = threshold_multiotsu(gray, classes=classes)

print("Optimal thresholds:", thresholds)

# Digitize image into regions
segmented = np.digitize(gray, bins=thresholds)

# Map regions to intensities for visualization
values = np.linspace(0, 255, classes).astype(np.uint8)
segmented_img = values[segmented]

# Display results
fig, ax = plt.subplots(1, 3, figsize=(12, 4))

ax[0].imshow(gray, cmap="gray")
ax[0].set_title("Grayscale")

ax[1].imshow(segmented, cmap="gray")
ax[1].set_title("Segmented (labels)")

ax[2].imshow(segmented_img, cmap="gray")
ax[2].set_title("Segmented (visualized)")

for a in ax:
    a.axis("off")

plt.tight_layout()
plt.show()
