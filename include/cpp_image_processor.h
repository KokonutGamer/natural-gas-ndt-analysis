#ifndef CPP_IMAGE_PROCESSOR_H_
#define CPP_IMAGE_PROCESSOR_H_

#include "include/image_processor.h"

/**
 * @class CppImageProcessor
 * @brief Concrete implementation of ImageProcessor utilizing C++ and OpenCV.
 * 
 * Processes images using native C++ routines, including morphological operations 
 * (dilation, opening, closing), median blurring, and custom histogram-based 
 * binary thresholding.
 */
class CppImageProcessor : public ImageProcessor
{
public:
    /**
     * @brief Executes the native C++ image processing pipeline.
     * 
     * Applies maximum and median filters, computes a threshold based on the bottom 
     * 5th percentile of the image histogram, applies a binary inverted mask, 
     * and performs morphological opening and closing.
     * 
     * @param image The input image as an OpenCV matrix.
     * @param method A string identifier for the specific processing algorithm to use (currently unused in the C++ implementation, provided for interface parity).
     * @return cv::Mat The processed binary image.
     */
    cv::Mat execute(const cv::Mat &image, const std::string method) const override;

    /**
     * @brief Retrieves the name of the C++ image processor.
     * 
     * @param method A string identifier for the specific processing algorithm to use.
     * @return std::string Always returns "C++ image processor".
     */
    std::string getName(const std::string method) const override;
};

#endif
