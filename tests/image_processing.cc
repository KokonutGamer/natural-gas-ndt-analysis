#include "include/image_processor.h"
#include "include/cpp_image_processor.h"
#include "include/py_image_processor.h"

#include <gtest/gtest.h>
#include <pybind11/embed.h>

namespace py = pybind11;

/**
 * TODO document fixture
 */
class ImageProcessingTest : public ::testing::Test
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

TEST_F(ImageProcessingTest, ExecuteReturnsNonEmptyMat)
{
    
}