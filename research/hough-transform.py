import cv2
import numpy as np

image = cv2.imread('images/scratch_2.tiff', cv2.IMREAD_GRAYSCALE)

# While sobel is direction, laplacian is not; it uses second-order derivatives instead
laplacian = cv2.Laplacian(image, cv2.CV_64F, ksize=3)
edges = cv2.convertScaleAbs(laplacian)
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 68, minLineLength=900, maxLineGap=1)

for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 3)

cv2.imshow("Hough Transform", image)

cv2.waitKey(0)
cv2.destroyAllWindows()