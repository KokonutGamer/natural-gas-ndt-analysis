#ifndef PY_IMAGE_PROCESSOR_H_
#define PY_IMAGE_PROCESSOR_H_

#include "image_processor.h"

#include <pybind11/embed.h>

namespace py = pybind11;

/**
 * @class PyImageProcessor
 * @brief Concrete implementation of ImageProcessor that delegates execution to
 * Python scripts.
 *
 * Uses pybind11 to embed a Python interpreter, load external Python plugins,
 * and dispatch image processing tasks to Python-based algorithms.
 */
class PyImageProcessor : public ImageProcessor {
public:
  /**
   * @brief Constructor that initializes the embedded Python interpreter.
   *
   * Acquires the Python Global Interpreter Lock (GIL), appends the project
   * script directory to the Python path, loads the plugins, and binds
   * the necessary Python functions and registries to C++ members.
   */
  PyImageProcessor();

  /**
   * @brief Destructor.
   *
   * Acquires the Python GIL before destruction to ensure safe cleanup
   * of Python objects.
   */
  ~PyImageProcessor() override;

  /**
   * @brief Dispatches the image processing task to the embedded Python
   * environment.
   *
   * @param image The input image as an OpenCV matrix.
   * @param method The string identifier of the Python algorithm to use.
   * @return cv::Mat The processed image returned from Python, or an empty
   * cv::Mat if an error occurs.
   */
  cv::Mat execute(const cv::Mat &image, const std::string method) const override;

  /**
   * @brief Retrieves the descriptive name of the specified Python algorithm.
   *
   * @param method The string identifier of the Python algorithm.
   * @return std::string The descriptive name, or "(Get name failed)" if an
   * error occurs.
   */
  std::string getName(const std::string method) const override;

  /**
   * @brief Retrieves a list of all processing methods registered in the Python
   * environment.
   *
   * @return std::vector<std::string> A vector containing the names of available
   * Python methods.
   */
  std::vector<std::string> getMethods() const;

private:
  /**
   * @brief Scoped interpreter guard that initializes and finalizes the Python
   * interpreter.
   *
   * Must be the first member to ensure it outlives all other Python objects.
   */
  py::scoped_interpreter guard;

  /**
   * @brief Pybind11 function object pointing to the Python execute dispatcher.
   */
  py::function dispatchExecute;

  /**
   * @brief Pybind11 function object pointing to the Python name dispatcher.
   */
  py::function dispatchName;

  /**
   * @brief Pybind11 dictionary containing the registered Python
   * methods/algorithms.
   */
  py::dict registry;
};

#endif
