import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

image = cv2.imread('images/BlackIron_OMS_corrosion-crack.tif', cv2.IMREAD_COLOR_RGB)

assert image is not None, "Could not load image"

image_float = image.astype(np.float32)
strict_gray_float : np.ndarray = np.mean(image_float, axis=2) # average over color channel
strict_gray : np.ndarray = strict_gray_float.astype(np.uint8) # unsigned char for gray (0-255)

rows, cols = strict_gray.shape

x = np.linspace(0, cols, cols)
y = np.linspace(0, rows, rows)
X, Y = np.meshgrid(x, y)

fig = plt.figure()
ax : Axes3D = fig.add_subplot(1, 1, 1, projection='3d')
surf = ax.plot_surface(X, Y, strict_gray, cmap='magma', linewidth=0, antialiased=False)

plt.show()
