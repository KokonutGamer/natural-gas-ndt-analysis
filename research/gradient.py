import cv2
import matplotlib.pyplot as plt

image = cv2.imread('images/Aluminum_Section_D1_MP.tif', cv2.IMREAD_GRAYSCALE)

assert image is not None, "Image could not be loaded. Check the file path."

# first order gradient (sobel operator)
sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)

sobelxy = cv2.Sobel(sobelx, cv2.CV_64F, 0, 1, ksize=5)
sobelyx = cv2.Sobel(sobely, cv2.CV_64F, 1, 0, ksize=5)

# second order gradient (laplacian)
laplacian = cv2.Laplacian(image, cv2.CV_64F, ksize=5)

fig, ax = plt.subplots(1, 4, figsize=(21, 13))

ax[0].imshow(image, cmap="gray")
ax[0].set_title("Grayscale")

ax[1].imshow(sobelxy, cmap="gray")
ax[1].set_title("Sobel X-Y")

ax[2].imshow(sobelyx, cmap="gray")
ax[2].set_title("Sobel Y-X")

ax[3].imshow(laplacian, cmap="gray")
ax[3].set_title("Laplacian")

plt.tight_layout()
plt.show()
