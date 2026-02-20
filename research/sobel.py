import cv2

# Load image in grayscale
img = cv2.imread('images/scratch_2.tiff', cv2.IMREAD_GRAYSCALE)

# Apply sobel kernel with kernel size 3 x 3
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3) # Highlights vertical edges 
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3) # Highlights horizontal edges

# NOTE: CV_64F denotes desired depth of the image (which includes negatives)

# Compute gradient magnitude
gradient_magnitude = cv2.magnitude(sobelx, sobely)

# Convert to uint8 (negatives become positive)
gradient_magnitude = cv2.convertScaleAbs(gradient_magnitude)

cv2.imshow("Sobel Edge Detection", gradient_magnitude)

cv2.waitKey(0)
cv2.destroyAllWindows()