import cv2
import numpy as np
from .abstract_pyprocessor import PyProcessor


class ContourPyProcessor(PyProcessor, key="cont"):
    """
    Concrete implementation of PyProcessor that isolates and draws contours.

    Registered in the PyProcessor registry under the key 'cont'.
    """

    def execute(self, image: np.ndarray) -> None:
        """
        Processes the image by filtering, thresholding, and drawing contours.

        Applies a pipeline of max and median filters, computes a binary
        threshold based on the bottom 5th percentile, performs morphological
        opening and closing, and finally finds and draws hierarchical contours
        with random colors onto a blanked canvas.

        Args:
            image (np.ndarray): The input OpenCV image array (modified in place).
        """
        # filter parameters
        n_filters = 4
        ksize = 3
        max_kernel = np.ones((ksize, ksize), np.uint8)

        for _ in range(n_filters):
            # max filter
            cv2.dilate(image, max_kernel, dst=image)

            # median filter
            cv2.medianBlur(image, ksize, dst=image)

        # find the threshold based on the bottom 5 percentile
        threshold_value = np.percentile(image, 5)

        # apply a binary mask on the image
        cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY_INV, dst=image)

        # morphological operation parameters
        n_morphs = 4
        ksize = 5
        morph_kernel = np.ones((ksize, ksize), np.uint8)

        for _ in range(n_morphs):
            # open
            cv2.morphologyEx(image, cv2.MORPH_OPEN, morph_kernel, dst=image)

            # close
            cv2.morphologyEx(image, cv2.MORPH_CLOSE, morph_kernel, dst=image)

        # find contours in the image and draw them
        cnts, hier = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        image.fill(0)
        for i in range(len(cnts)):
            color = np.random.randint(0, 256)
            cv2.drawContours(image, cnts, i, color, 3, cv2.LINE_8, hier, 0)

    def get_name(self) -> str:
        """
        Retrieves the name of the contour image processor.

        Returns:
            str: "Contour Python image processor"
        """
        return "Contour Python image processor"
