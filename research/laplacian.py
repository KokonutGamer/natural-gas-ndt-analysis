import cv2

image = cv2.imread('images/scratch_2.tiff', cv2.IMREAD_GRAYSCALE)

# While sobel is direction, laplacian is not; it uses second-order derivatives instead
laplacian = cv2.Laplacian(image, cv2.CV_64F, ksize=3)

laplacian_abs = cv2.convertScaleAbs(laplacian)

cv2.imshow("Laplacian Edge Detection", laplacian_abs)

cv2.waitKey(0)
cv2.destroyAllWindows()