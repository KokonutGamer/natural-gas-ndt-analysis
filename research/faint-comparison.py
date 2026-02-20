import cv2
import numpy as np
import matplotlib.pyplot as plt

def process(image_number):
    """
    ========== IMAGE PROCESSING ==========
    """
    # Read as 8-bit grayscale
    image = cv2.imread(f'images/faint_{image_number}.tiff', cv2.IMREAD_GRAYSCALE)
    if image is None:
        return np.zeros((480, 640, 3), dtype=np.uint8) # Return black if file missing

    n_filters = 4
    ksize = 3
    max_kernel = np.ones((ksize, ksize), np.uint8)

    for i in range(n_filters):
        image = cv2.dilate(image, max_kernel)
        image = cv2.medianBlur(image, ksize)

    # Threshold based on the bottom 5 percentile
    threshold_value = np.percentile(image, 5)
    (_, image) = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY_INV)

    # Open and close the image
    n_morphs = 4
    ksize = 5
    morph_kernel = np.ones((ksize, ksize), np.uint8)

    for i in range(n_morphs):
        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, morph_kernel)
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, morph_kernel)

    """
    ========== HOUGH TRANSFORM ==========
    """
    # Prepare color canvas
    line_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    lines = cv2.HoughLinesP(image, 1, np.pi / 180, 30, minLineLength=50, maxLineGap=4)
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Red lines (BGR for OpenCV)
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 3)

    # Convert BGR to RGB for Matplotlib and return
    return cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB)

"""
========== MULTI-IMAGE PLOTTING (2 ROWS) ==========
"""
# Create a 2x4 grid (8 slots total for 7 images)
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

# Flatten the axes array from 2D (2,4) to 1D (8,) for easier looping
axes_flat = axes.flatten()

for i in range(1, 8):
    result_img = process(i)
    
    # Plot on the corresponding flattened axis
    ax = axes_flat[i-1]
    ax.imshow(result_img)
    ax.set_title(f"Image {i}", fontsize=14)
    ax.axis('off')

# Hide the 8th (empty) subplot
axes_flat[-1].axis('off')

plt.tight_layout()
plt.show()