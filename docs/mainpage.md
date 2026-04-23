# NDT Image Analyzer Developer Documentation

Welcome to the developer documentation for the **Non-Destructive Testing (NDT) Image Analyzer**. 

This software is designed to run on the Raspberry Pi 5 to process images and detect micro-anomalies such as microcracks and micropits. It utilizes a highly flexible, hybrid C++/Python architecture, allowing developers the ability to balance performance with rapid prototyping.

## System Architecture

The project is built around a polymorphic core using C++ and OpenCV, while leveraging `pybind11` to embed a fully functioning Python interpreter. This allows the system to execute either native C++ code or dynamic Python scripts without changing the core driver application.

### The C++ Core
The entry point of the application (`driver.cc`) parses command-line arguments using `cxxopts` and dynamically instantiates a subclass of the `ImageProcessor` base class.

* `ImageProcessor`: The abstract interface defining the `execute()` and `getName()` methods.
* `CppImageProcessor`: A native implementation executing image processing entirely in C++ using OpenCV. It implements a Filter-Mask-Morph (FMM) pipeline natively.
* `PyImageProcessor`: A specialized implementation that boots up an embedded Python interpreter, acquires the Global Interpreter Lock (GIL), and dispatches image processing tasks to external Python scripts.

### The Python Plugin System
When `PyImageProcessor` is instantiated, it automatically executes `plugin_loader.py`. This script dynamically scans the `scripts/` directory for any file ending in `*processor.py` and imports it.

All Python algorithms must inherit from the `PyProcessor` abstract base class. When a subclass is defined, the `__init_subclass__` hook automatically registers it into an internal dictionary using a designated `key`.

Currently implemented Python algorithms include:
* [FMMorphPyProcessor](@ref fmmorph_pyprocessor.FMMorphPyProcessor) (Key: `fmm`): Replicates the C++ Filter-Mask-Morph pipeline in Python.
* [ContourPyProcessor](@ref contour_pyprocessor.ContourPyProcessor) (Key: `cont`): Processes the image and draws hierarchical bounding contours around detected anomalies.

### Zero-Copy Memory Bridging
A crucial component of this architecture is the `cvmat_caster.hpp`. This file defines a custom `pybind11` type caster that maps C++ `cv::Mat` objects directly to Python `numpy.ndarray` objects. 

When an image is passed from `PyImageProcessor` to a Python script, it is passed by reference (zero-copy where possible) and modified **in-place** by the Python script, avoiding expensive deep memory copies between the two languages.

---

## Developer Guide: Adding a New Python Algorithm

Thanks to the dynamic plugin loader, you can add new image processing algorithms without recompiling the C++ core.

1. **Create a new file** in the `scripts/` directory named `<your_algorithm>_processor.py`.
2. **Inherit from `PyProcessor`** and provide a unique string key.
3. **Implement the `execute` and `get_name` methods.**
4. **Modify the image in-place** (the image is passed as a `numpy.ndarray`).

### Example:
```Python
import cv2
import numpy as np
from abstract_pyprocessor import PyProcessor

class EdgeDetectionPyProcessor(PyProcessor, key='edges'):
    
    def execute(self, image: np.ndarray) -> None:
        # Apply a Canny edge detector
        edges = cv2.Canny(image, 100, 200)
        
        # Copy the processed data back into the original array 
        # to ensure the C++ caller sees the changes
        np.copyto(image, edges)
        
    def get_name(self) -> str:
        return "Canny Edge Detector"
```
