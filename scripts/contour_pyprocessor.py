import cv2
import numpy as np
from abstract_pyprocessor import PyProcessor

"""
TODO document 
"""
class ContourPyProcessor(PyProcessor, key='cont'):
    """
    TODO document execute method
    """
    def execute(self, image: np.ndarray) -> None:
        # filter parameters
        n_filters = 4
        ksize = 3
        max_kernel = np.ones((ksize, ksize), np.uint8)
        
        for i in range(n_filters):
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
        
        for i in range(n_morphs):
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
            

    """
    TODO document name method
    """
    def get_name(self) -> str:
        return "Contour Python image processor"