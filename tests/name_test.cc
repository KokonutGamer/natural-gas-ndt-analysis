#include <gtest/gtest.h>
#include <pybind11/embed.h>

#include "include/image_processor.h"
#include "include/cpp_image_processor.h"
#include "include/py_image_processor.h"

namespace py = pybind11;

/**
 * TODO document fixture
 */
class NameTest : public ::testing::Test
{
protected:
    static void SetUpTestSuite()
    {
        if (!Py_IsInitialized())
        {
            py::initialize_interpreter();
        }
    }
};

TEST_F(NameTest, CppImageProcessorHasCorrectName)
{
    // Arrange
    auto processor = std::make_unique<CppImageProcessor>();

    // Act
    std::string name = processor->getName();

    // Assert
    EXPECT_EQ(name, "C++ image processor");
}

TEST_F(NameTest, PyImageProcessorHasCorrectName)
{
    // Arrange
    auto processor = std::make_unique<PyImageProcessor>();

    // Act
    std::string name = processor->getName();

    // Assert
    EXPECT_EQ(name, "Python image processor");
}