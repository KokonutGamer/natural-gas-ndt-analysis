# Requiers CMake version 3.23+
cmake_minimum_required(VERSION 3.23)

# Project metadata
project(NDTImageProcessor LANGUAGES CXX)

# C++ configuration details
set(CMAKE_CXX_STANDARD 20)              # C++ version 20
set(CMAKE_CXX_STANDARD_REQUIRED ON)     # C++ version required
set(CMAKE_CXX_VISIBILITY_PRESET hidden) # Set default visibility to hidden
set(CMAKE_VISIBILITY_INLINES_HIDDEN 1)  # Ensure that inline functions also have hidden visibility

# Used for C/C++ IntelliSense
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Dependencies
include(FetchContent)

# Pybind11
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY  https://github.com/pybind/pybind11.git
    GIT_TAG         f5fbe867d2d26e4a0a9177a51f6e568868ad3dc8 # release-3.0.1
    GIT_SHALLOW     TRUE
    GIT_PROGRESS    TRUE
)
FetchContent_MakeAvailable(pybind11)

# Google Test
FetchContent_Declare(
    googletest
    GIT_REPOSITORY  https://github.com/google/googletest.git
    GIT_TAG         52eb8108c5bdec04579160ae17225d66034bd723 # release-1.17.0
    GIT_SHALLOW     TRUE
    GIT_PROGRESS    TRUE
)

# DO NOT USE STATIC RUNTIME ON WINDOWS
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Google Benchmark
FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY  https://github.com/google/benchmark.git
    GIT_TAG         192ef10025eb2c4cdd392bc502f0c852196baa48 # release-1.9.5
    GIT_SHALLOW     TRUE
    GIT_PROGRESS    TRUE
)
FetchContent_MakeAvailable(googlebenchmark)

# Tracy profiler
FetchContent_Declare(
    tracy
    GIT_REPOSITORY  https://github.com/wolfpld/tracy.git
    GIT_TAG         05cceee0df3b8d7c6fa87e9638af311dbabc63cb # release-0.13.1
    GIT_SHALLOW     TRUE
    GIT_PROGRESS    TRUE
)
FetchContent_MakeAvailable(tracy)

# Python configuration
set(Python3_FIND_VIRTUALENV FIRST)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
message(STATUS "CMake found Python at: ${Python3_EXECUTABLE}")

# OpenCV configuration
find_package(OpenCV REQUIRED)

# Core library - logic that does not contain main
add_library(imgproc_core
    src/cpp_image_processor.cc
    src/py_image_processor.cc
)

# Define include directories for the library
target_include_directories(imgproc_core PUBLIC .) # root for relative imports

# Link Python, Pybind, and OpenCV to the core library
target_link_libraries(imgproc_core PUBLIC pybind11::embed ${OpenCV_LIBS})

# Application
add_executable(imgproc
    src/driver.cc
)

# Link core logic as a library
target_link_libraries(imgproc PRIVATE imgproc_core)

# Testing with Google Test
enable_testing()

add_executable(unit_tests
    tests/name_test.cc
)

target_link_libraries(unit_tests PRIVATE
    imgproc_core
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(unit_tests)

# Benchmarking with Google Benchmark
add_executable(benchmarks
    benchmark/image_processing_benchmark.cc
)

target_link_libraries(benchmarks PRIVATE
    imgproc_core
    benchmark::benchmark_main
)

# Install the main binary
install(TARGETS imgproc DESTINATION bin)

# Determine the python version for the installation path
set(PYTHON_INSTALL_DIR "lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages")

# Install scripts to the relative path
install(DIRECTORY scripts DESTINATION ${PYTHON_INSTALL_DIR})

# Compile macro definitions
target_compile_definitions(imgproc_core PRIVATE PROJECT_ROOT_DIR=\"${CMAKE_SOURCE_DIR}\")
target_compile_definitions(benchmarks PRIVATE PROJECT_ROOT_DIR=\"${CMAKE_SOURCE_DIR}\")
