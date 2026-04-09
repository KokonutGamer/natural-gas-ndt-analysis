import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('images/large_pit_1.tiff', cv2.IMREAD_COLOR)

if image is None:
    raise FileNotFoundError("Image could not be loaded. Check the file path.")

image_float = image.astype(np.float32)
strict_gray_float = np.mean(image_float, axis=2) # along color channel
strict_gray = strict_gray_float.astype(np.uint8)

h, w = strict_gray.shape[:2]
center_y, center_x = h // 2, w // 2
half_size = min(h, w) // 2

og = strict_gray[center_y - half_size:center_y + half_size, center_x - half_size:center_x + half_size]

median_image = cv2.medianBlur(og, 3)
gaussian_image = cv2.GaussianBlur(og, (0, 0), 9.5) # auto size kernel
filtered_image = cv2.GaussianBlur(median_image, (0, 0), 10) # auto size kernel

fig, ax = plt.subplots(1, 4, figsize=(21, 13))

ax[0].imshow(og, cmap="gray")
ax[0].set_title("Grayscale")

ax[1].imshow(median_image, cmap="gray")
ax[1].set_title("Median Blur")

ax[2].imshow(gaussian_image, cmap="gray")
ax[2].set_title("Gaussian Blur")

ax[3].imshow(filtered_image, cmap="gray")
ax[3].set_title("Median Blur then Gaussian Blur")

plt.tight_layout()
plt.show()
