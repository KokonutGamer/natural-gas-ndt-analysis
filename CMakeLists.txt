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

# Enable the Tracy profiler
option(TRACY_ENABLE "Enable Tracy profiling" ON)

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

set(CORE_SRCS src/cpp_image_processor.cc src/py_image_processor.cc)

# Clean core library without embedded profiler
add_library(imgproc_core ${CORE_SRCS})
target_include_directories(imgproc_core PUBLIC .) # root for relative imports
target_include_directories(imgproc_core PUBLIC ${tracy_SOURCE_DIR}/public)
target_link_libraries(imgproc_core PUBLIC pybind11::embed ${OpenCV_LIBS})
target_compile_definitions(imgproc_core PRIVATE PROJECT_ROOT_DIR=\"${CMAKE_SOURCE_DIR}\")

# Profiled core library for main executable
add_library(imgproc_core_tracy ${CORE_SRCS})
target_include_directories(imgproc_core_tracy PUBLIC .) # root for relative imports
target_link_libraries(imgproc_core_tracy PUBLIC pybind11::embed ${OpenCV_LIBS} TracyClient)
target_compile_definitions(imgproc_core_tracy PUBLIC TRACY_ENABLE)
target_compile_definitions(imgproc_core_tracy PRIVATE PROJECT_ROOT_DIR=\"${CMAKE_SOURCE_DIR}\")
find_package(Threads REQUIRED)
target_link_libraries(imgproc_core_tracy PUBLIC Threads::Threads ${CMAKE_DL_LIBS})


# Main application with profiler (links with profiled core library)
add_executable(imgproc src/driver.cc)
target_link_libraries(imgproc PRIVATE imgproc_core_tracy)

# Testing with Google Test
enable_testing()

# Unit tests with Google test (links with clean core library)
add_executable(unit_tests tests/name_test.cc)
target_link_libraries(unit_tests PRIVATE imgproc_core GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(unit_tests)

# Benchmarks with Google benchmark
add_executable(benchmarks benchmark/image_processing_benchmark.cc)
target_link_libraries(benchmarks PRIVATE imgproc_core benchmark::benchmark_main)
target_compile_definitions(benchmarks PRIVATE PROJECT_ROOT_DIR=\"${CMAKE_SOURCE_DIR}\")

# Install the main binary
install(TARGETS imgproc DESTINATION bin)

# Determine the python version for the installation path
set(PYTHON_INSTALL_DIR "lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages")

# Install scripts to the relative path
install(DIRECTORY scripts DESTINATION ${PYTHON_INSTALL_DIR})
