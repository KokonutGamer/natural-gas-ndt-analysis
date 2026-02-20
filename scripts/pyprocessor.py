import cv2
import numpy as np

"""
TODO document execute method
"""
def execute(image: np.ndarray) -> np.ndarray:
    # copy the image to a new image
    processed_image = image.copy()
    
    # filter parameters
    n_filters = 4
    ksize = 3
    max_kernel = np.ones((ksize, ksize), np.uint8)
    
    for i in range(n_filters):
        # max filter
        processed_image = cv2.dilate(processed_image, max_kernel)
        
        # median filter
        processed_image = cv2.medianBlur(processed_image, ksize)
    
    # find the threshold based on the bottom 5 percentile
    threshold_value = np.percentile(processed_image, 5)
    
    # apply a binary mask on the image
    (_, processed_image) = cv2.threshold(processed_image, threshold_value, 255, cv2.THRESH_BINARY_INV)
    
    # morphological operation parameters
    n_morphs = 4
    ksize = 5
    morph_kernel = np.ones((ksize, ksize), np.uint8)
    
    for i in range(n_morphs):
        # open
        processed_image = cv2.morphologyEx(processed_image, cv2.MORPH_OPEN, morph_kernel)
        
        # close
        processed_image = cv2.morphologyEx(processed_image, cv2.MORPH_CLOSE, morph_kernel)
        
    return processed_image

"""
TODO document name method
"""
def get_name():
    return "Python image processor"