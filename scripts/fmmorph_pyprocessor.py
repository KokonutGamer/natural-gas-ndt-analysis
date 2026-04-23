import cv2
import numpy as np
from abstract_pyprocessor import PyProcessor


class FMMorphPyProcessor(PyProcessor, key="fmm"):
    """
    Concrete implementation of PyProcessor executing an FMM pipeline.

    Applies Filter, Mask, and Morph (FMM) operations to the image.
    Registered in the PyProcessor registry under the key 'fmm'.
    """

    def execute(self, image: np.ndarray) -> None:
        """
        Executes the FMM (Filter-Mask-Morph) processing pipeline in place.

        Applies max and median filtering, creates an inverted binary mask
        based on the bottom 5th percentile, and performs morphological
        opening and closing to isolate specific features.

        Args:
            image (np.ndarray): The input OpenCV image array (modified in place).
        """
        # image processing in place (assumes the image passed into the function
        # is a deep-copy)

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

    def get_name(self) -> str:
        """
        Retrieves the name of the FMM image processor.

        Returns:
            str: "Filter-Mask-Morph (FMM) Python image processor"
        """
        return "Filter-Mask-Morph (FMM) Python image processor"
