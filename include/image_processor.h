#ifndef IMAGE_PROCESSOR_H_
#define IMAGE_PROCESSOR_H_

#include <opencv2/opencv.hpp>
#include <pybind11/embed.h>

namespace py = pybind11;

/**
 * TODO document image processor
 */
class ImageProcessor
{
public:
    /**
     * TODO document constructor
     */
    ImageProcessor() = default;

    /**
     * TODO document copy/move deletion
     */
    ImageProcessor(const ImageProcessor &) = delete;
    ImageProcessor &operator=(const ImageProcessor &) = delete;
    ImageProcessor(const ImageProcessor &&) = delete;
    ImageProcessor &operator=(const ImageProcessor &&) = delete;

    /**
     * TODO document destructor
     */
    virtual ~ImageProcessor() = default;

    /**
     * TODO document execute method
     */
    virtual cv::Mat execute(const cv::Mat &image, const std::string method) const = 0;

    /**
     * TODO document name method
     */
    virtual std::string getName(const std::string method) const = 0;
};

#endif