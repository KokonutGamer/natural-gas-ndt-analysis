import cv2
import numpy as np
import matplotlib.pyplot as plt

# Steps to reproduce:
# 1. Convert to 8-bit grayscale image
# 2. Repeat the following some number of times:
#   a. Apply a max filter of radius 2 px
#   b. Apply a median filter of radius 2 px
# 3. Threshold based on the bottom 5% of gray values
# 4. Invert the image for hough transform
# 5. (Needs further testing) Apply a series of openings and closings
# 6. Ready for hough transform!

"""
========== IMAGE PROCESSING ==========
"""

# Read as 8-bit grayscale
image = cv2.imread('images/two_vertical_and_horizontal.tiff', cv2.IMREAD_GRAYSCALE)
# image = cv2.imread('images/faint_pit.tiff', cv2.IMREAD_GRAYSCALE)
# image = cv2.imread('images/pit.tiff', cv2.IMREAD_GRAYSCALE)


og = image.copy()

n_filters = 4
ksize = 3
max_kernel = np.ones((ksize, ksize), np.uint8)

for i in range(n_filters):
    # max filter
    image = cv2.dilate(image, max_kernel)
    
    # median filter
    image = cv2.medianBlur(image, ksize)

filtered_image = image.copy()

# Find the threshold based on the bottom 5 percentile
threshold_value = np.percentile(image, 5)

# Binary mask
(_, image) = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY_INV)

binary = image.copy()

# Open and close the image
n_morphs = 4
ksize = 5
morph_kernel = np.ones((ksize, ksize), np.uint8)

for i in range(n_morphs):
    # open
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, morph_kernel)
    
    # close
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, morph_kernel)

processed_binary = image.copy()

"""
========== HOUGH TRANSFORM ==========
"""
line_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
lines = cv2.HoughLinesP(image, 1, np.pi / 180, 30, minLineLength=50, maxLineGap=4)
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 3)

"""
========== PLOTTING ==========
"""
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(2, 4)

# Top Row
ax1 = fig.add_subplot(gs[0, 0])
ax1.imshow(og, cmap="gray")
ax1.set_title("Grayscale")

ax2 = fig.add_subplot(gs[0, 1])
ax2.imshow(filtered_image, cmap="gray")
ax2.set_title(f'After {n_filters} max + median')

ax3 = fig.add_subplot(gs[0, 2])
ax3.imshow(binary, cmap="gray")
ax3.set_title(f'Binary (t = {threshold_value})')

ax4 = fig.add_subplot(gs[0, 3])
ax4.imshow(processed_binary, cmap="gray")
ax4.set_title("Fully processed")

ax5 = fig.add_subplot(gs[1, :])
ax5.imshow(cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB))
ax5.set_title("Hough lines")

plt.tight_layout()
plt.show()