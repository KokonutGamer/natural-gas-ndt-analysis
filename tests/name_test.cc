#include "include/image_processor.h"
#include "include/cpp_image_processor.h"
#include "include/py_image_processor.h"

#include <gtest/gtest.h>
#include <pybind11/embed.h>

// processors only ever created once
static std::unique_ptr<CppImageProcessor> cppImageProcessor = std::make_unique<CppImageProcessor>();
static std::unique_ptr<PyImageProcessor> pyImageProcessor = std::make_unique<PyImageProcessor>();

TEST(NameTest, CppImageProcessorHasCorrectName)
{
    // Act
    std::string name = cppImageProcessor->getName();

    // Assert
    EXPECT_EQ(name, "C++ image processor");
}

TEST(NameTest, PyImageProcessorHasCorrectName)
{
    // Act
    std::string name = pyImageProcessor->getName();

    // Assert
    EXPECT_EQ(name, "Python image processor");
}