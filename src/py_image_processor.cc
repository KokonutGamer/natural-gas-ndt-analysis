#include "include/cvmat_caster.hpp"
#include "include/py_image_processor.h"

#include <iostream>
#include <pybind11/stl.h>

PyImageProcessor::PyImageProcessor()
{
    // acquire the lock on the python gil
    py::gil_scoped_acquire acquire;

    // import the sys module and add project root directory
    py::module_ sys = py::module_::import("sys");
    sys.attr("path").attr("append")(PROJECT_ROOT_DIR "/scripts");

    // import all concrete implementations of the abstract base class
    py::module_::import("plugin_loader");

    // setup abstract base class bindings
    py::object abc = py::module_::import("abstract_pyprocessor").attr("PyProcessor");
    this->dispatchExecute = abc.attr("dispatch_execute");
    this->dispatchName = abc.attr("dispatch_name");

    // store the registry for benchmark information
    this->registry = abc.attr("get_registry")();
}

PyImageProcessor::~PyImageProcessor()
{
    // acquire the lock on the python gil
    py::gil_scoped_acquire acquire;
}

cv::Mat PyImageProcessor::execute(const cv::Mat &image, const std::string method) const
{
    // acquire the lock on the python gil
    py::gil_scoped_acquire acquire;

    // deep copy of image
    cv::Mat processedImage = image.clone();
    try
    {
        this->dispatchExecute(method, processedImage);
        return processedImage;
    }
    catch (const std::exception &e)
    {
        // return an empty cv mat
        std::cerr << e.what() << '\n';
        return cv::Mat();
    }
}

std::string PyImageProcessor::getName(const std::string method) const
{
    // acquire the lock on the python gil
    py::gil_scoped_acquire acquire;
    try
    {
        return this->dispatchName(method).cast<std::string>();
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
        return "(Get name failed)";
    }
}

std::vector<std::string> PyImageProcessor::getMethods() const
{
    // acquire the lock on the python gil
    py::gil_scoped_acquire acquire;
    try
    {
        return py::list(this->registry.attr("keys")()).cast<std::vector<std::string>>();
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
        return {};
    }
}