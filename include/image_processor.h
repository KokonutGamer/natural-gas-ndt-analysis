#ifndef IMAGE_PROCESSOR_H_
#define IMAGE_PROCESSOR_H_

#include <opencv2/opencv.hpp>
#include <pybind11/embed.h>

namespace py = pybind11;

/**
 * @class ImageProcessor
 * @brief Abstract base class for image processing algorithms.
 *
 * Provides a standard interface for executing image processing routines
 * and retrieving metadata, allowing for polymorphic usage of different
 * backend implementations.
 */
class ImageProcessor {
public:
  /**
   * @brief Default constructor.
   */
  ImageProcessor() = default;

  /**
   * @brief Deleted copy and move semantics.
   *
   * Ensures processors are unique and non-shareable.
   */
  ImageProcessor(const ImageProcessor &) = delete;
  ImageProcessor &operator=(const ImageProcessor &) = delete;
  ImageProcessor(const ImageProcessor &&) = delete;
  ImageProcessor &operator=(const ImageProcessor &&) = delete;

  /**
   * @brief Virtual destructor.
   *
   * Ensures derived classes are properly destroyed.
   */
  virtual ~ImageProcessor() = default;

  /**
   * @brief Executes the image processing algorithm on the provided image.
   *
   * @param image The input image as an OpenCV matrix.
   * @param method A string identifier for the specific processing algorithm to
   * use.
   * @return cv::Mat The newly processed image.
   */
  virtual cv::Mat execute(const cv::Mat &image, const std::string method) const = 0;

  /**
   * @brief Retrieves the descriptive name of the image processor or specific
   * method.
   *
   * @param method A string identifier for the specific processing algorithm to
   * use.
   * @return std::string The descriptive name of the underlying algorithm.
   */
  virtual std::string getName(const std::string method) const = 0;
};

#endif
