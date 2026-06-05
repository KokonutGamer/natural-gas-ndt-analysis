#ifndef CVMAT_CASTER_HPP_
#define CVMAT_CASTER_HPP_

#include <opencv2/core/core.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace pybind11 {
namespace detail {
/**
 * @struct type_caster<cv::Mat>
 * @brief Custom pybind11 type caster to convert between OpenCV cv::Mat and
 * Python numpy.ndarray.
 */
template <> struct type_caster<cv::Mat> {
public:
  /**
   * @brief Defines the Python type name as "numpy.ndarray" and sets up internal
   * value storage.
   */
  PYBIND11_TYPE_CASTER(cv::Mat, _("numpy.ndarray"));

  /**
   * @brief Converts a Python numpy.ndarray into a C++ cv::Mat.
   *
   * Reads the buffer information of the provided Python array to determine
   * dimensions, strides, channels, and data type (e.g., 8-bit unsigned,
   * 32-bit floating point). Wraps the data in a cv::Mat without copying.
   *
   * @param src The Python object handle representing the source array.
   * @param convert Unused flag indicating whether implicit conversion is
   * allowed.
   * @return true If the Python object was successfully cast to a cv::Mat.
   * @return false If the object is not a valid numpy.ndarray or has an
   * unsupported type.
   */
  bool load(handle src, bool convert) {
    // cannot convert an object that is not a numpy.ndarray
    if (!isinstance<array>(src))
      return false;

    // borrow python's reference to the object and reinterpret it into a
    // numpy.ndarray
    array b = reinterpret_borrow<array>(src);

    // request the buffer information of the array
    // this information contains the metadata pertaining to the array
    // including dimensions, shape, type information, etc.
    buffer_info info = b.request();

    int ndims = info.ndim;
    int dtype;
    if (info.format == format_descriptor<unsigned char>::format()) {
      dtype = CV_8U; // 8-bit unsigned
    } else if (info.format == format_descriptor<int>::format()) {
      dtype = CV_32S; // 32-bit signed
    } else if (info.format == format_descriptor<float>::format()) {
      dtype = CV_32F; // 32-bit floating point
    } else if (info.format == format_descriptor<double>::format()) {
      dtype = CV_64F; // 64-bit double precision
    } else {
      return false; // unsupported type detected
    }

    // set the number of channels based on if the third dimension exists
    int channels = 1;
    if (ndims == 3) {
      // save to lookup using square brackets since checked dimensions first
      channels = static_cast<int>(info.shape[2]);
    }

    // cv::Mat view (zero-copy)
    int rows = static_cast<int>(info.shape[0]);
    int cols = static_cast<int>(info.shape[1]);
    int type = CV_MAKE_TYPE(dtype, channels);

    // store the mat into the type_caster value property
    value = cv::Mat(rows, cols, type, info.ptr, info.strides[0]);
    return true;
  }

  /**
   * @brief Converts a C++ cv::Mat into a Python numpy.ndarray.
   *
   * Maps the OpenCV matrix depths and dimensions to the corresponding
   * Python format descriptor strings, shapes, and strides. Attaches a
   * capsule to ensure the C++ matrix is correctly deallocated when the
   * Python object is garbage collected.
   *
   * @param m The OpenCV matrix (cv::Mat) to convert.
   * @param return_value_policy Policy regulating how memory should be managed.
   * @param defval Default value handle (unused).
   * @return handle A Python handle representing the newly created
   * numpy.ndarray.
   * @throws std::runtime_error If the cv::Mat has an unsupported depth/type.
   */
  static handle cast(const cv::Mat &m, return_value_policy, handle defval) {
    // 8-bit unsigned as the default format descriptor
    std::string format = format_descriptor<unsigned char>::format();

    // get the size of one value in one channel (ignores number of channels)
    size_t elemSize = m.elemSize1();

    // map OpenCV types to Python format strings

    // currently, we probably don't need most of these checks as
    // the camera should be set to use 8-bit with three channels;
    // however, it might be a good idea to leave these as-is in
    // case a future iteration decides to change the camera
    // settings
    int depth = m.depth();
    if (depth == CV_8U) {
      format = format_descriptor<unsigned char>::format();
    } else if (depth == CV_32S) {
      format = format_descriptor<int>::format();
    } else if (depth == CV_32F) {
      format = format_descriptor<float>::format();
    } else if (depth == CV_64F) {
      format = format_descriptor<double>::format();
    } else {
      throw std::runtime_error("Unsupported cv::Mat type for conversion");
    }

    // define shape metadata using rows and columns
    std::vector<ssize_t> shape = {m.rows, m.cols};

    // define strides based on steps for rows and columns

    // strides help numpy determine how many elements to pass to get to the next
    // element along the specified axis
    std::vector<ssize_t> strides = {static_cast<ssize_t>(m.step[0]), static_cast<ssize_t>(m.step[1])};

    // add another element to the shape and stride tuples if multiple color
    // channels exist
    if (m.channels() > 1) {
      shape.push_back(m.channels());
      strides.push_back(elemSize);
    }

    // copy m's header info while incrementing the shared pointer in m
    cv::Mat *safeMat = new cv::Mat(m);

    // create a capsule to be attached for cleanup in Python's GC
    capsule cleanup(safeMat, [](void *ptr) {
      // reinterpret the void pointer from Python into a cv::Mat
      cv::Mat *matToDelete = reinterpret_cast<cv::Mat *>(ptr);

      // deallocate Python's cv::Mat instance when cleaning up
      delete matToDelete;
    });

    // create a numpy.ndarray that shares memory with cv::Mat
    // returns a raw handle to original Python object
    return array(buffer_info(m.data,                   // pointer to buffer
                             elemSize,                 // size of one element
                             format,                   // Python struct-style format descriptor
                             m.channels() > 1 ? 3 : 2, // number of dimensions
                             shape,                    // buffer dimensions
                             strides                   // strides for each index
                             ),
                 cleanup)
        .release();
  }
};
} // namespace detail
}; // namespace pybind11

#endif
