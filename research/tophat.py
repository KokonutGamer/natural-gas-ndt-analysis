import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image using OpenCV
img = cv2.imread("images/scratch_2.tiff")

assert img is not None, "Image could not be loaded. Check the file path."

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Kernel for top hat
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# Top hat image
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))

ax[0].imshow(gray, cmap="gray")
ax[0].set_title("Grayscale")

ax[1].imshow(tophat, cmap="gray")
ax[1].set_title("Tophat")

plt.tight_layout()
plt.show()