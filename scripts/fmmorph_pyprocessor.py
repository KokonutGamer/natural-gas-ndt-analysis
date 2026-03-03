import cv2
import numpy as np
from abstract_pyprocessor import PyProcessor

"""
TODO document FMMorphPyProcessor concrete class
"""
class FMMorphPyProcessor(PyProcessor, key='fmm'):
    """
    TODO document execute method
    """
    def execute(self, image: np.ndarray) -> None:
        # image processing in place (assumes the image passed into the function
        # is a deep-copy)
        
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
        
    
    """
    TODO document name method
    """
    def get_name(self) -> str:
        return "Filter-Mask-Morph (FMM) Python image processor"
