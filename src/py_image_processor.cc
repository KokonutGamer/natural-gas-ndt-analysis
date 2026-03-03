#include "include/cvmat_caster.hpp"
#include "include/py_image_processor.h"

#include <iostream>

PyImageProcessor::PyImageProcessor()
{
    // acquire the lock on the python gil
    py::gil_scoped_acquire acquire;

    // import the sys module and add project root directory
    py::module_ sys = py::module_::import("sys");
    sys.attr("path").attr("append")(PROJECT_ROOT_DIR "/scripts");

    // import the python module and functions
    pyProcessor = py::module_::import("pyprocessor");
    executeFunction = pyProcessor.attr("execute");
    getNameFunction = pyProcessor.attr("get_name");

    // import the abstract base class for python implementations
    py::module_::import("fmmorph_pyprocessor");
    this->registry = py::module_::import("abstract_pyprocessor").attr("PyProcessor").attr("get_registry")();
}

PyImageProcessor::~PyImageProcessor()
{
    // acquire the lock on the python gil
    py::gil_scoped_acquire acquire;
}

cv::Mat PyImageProcessor::execute(const cv::Mat &image) const
{
    // acquire the lock on the python gil
    py::gil_scoped_acquire acquire;

    // deep copy of image
    cv::Mat processedImage = image.clone();
    try
    {
        this->registry["ffm"].attr("execute")(this->registry["ffm"], processedImage);
        return processedImage;
    }
    catch (const std::exception &e)
    {
        // return an empty cv mat
        std::cerr << e.what() << '\n';
        return cv::Mat();
    }
}

std::string PyImageProcessor::getName() const
{
    // acquire the lock on the python gil
    py::gil_scoped_acquire acquire;
    try
    {
        return getNameFunction().cast<std::string>();
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
        return "Get name failed";
    }
}