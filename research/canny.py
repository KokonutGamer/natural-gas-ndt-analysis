import cv2

image = cv2.imread('images/scratch_2.tiff', cv2.IMREAD_GRAYSCALE)

# Apply Gaussian blur first to reduce noise
blur = cv2.GaussianBlur(image, (5, 5), 1.4)

cv2.imshow("Gaussian Blur", blur)

# Canny Edge Detector
edges = cv2.Canny(blur, threshold1=100, threshold2=200)

cv2.imshow("Canny Edge Detection", edges)

cv2.waitKey(0)
cv2.destroyAllWindows()