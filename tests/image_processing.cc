#include "include/cpp_image_processor.h"
#include "include/image_processor.h"
#include "include/py_image_processor.h"

#include <algorithm>
#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

// processors only ever created once
static std::unique_ptr<CppImageProcessor> cppImageProcessor = std::make_unique<CppImageProcessor>();
static std::unique_ptr<PyImageProcessor> pyImageProcessor = std::make_unique<PyImageProcessor>();

// ==========================================
// C++ Processor Tests
// ==========================================

TEST(ImageProcessingTest, CppExecuteReturnsNonEmptyImage) {
  // Arrange
  cv::Mat dummyImage = cv::Mat::ones(10, 10, CV_8UC1) * 255;

  // Act
  cv::Mat result = cppImageProcessor->execute(dummyImage, "");

  // Assert
  // verify matrix is valid and dimensions match
  EXPECT_FALSE(result.empty()) << "The C++ processor should return a valid matrix.";
  EXPECT_EQ(result.rows, dummyImage.rows) << "Height should remain the same.";
  EXPECT_EQ(result.cols, dummyImage.cols) << "Width should remain the same.";
}

// ==========================================
// Python Processor Tests
// ==========================================

TEST(ImageProcessingTest, PyGetMethodsReturnsRegisteredAlgorithms) {
  // Act
  std::vector<std::string> methods = pyImageProcessor->getMethods();

  // Assert
  EXPECT_FALSE(methods.empty()) << "Python processor should have registered methods.";

  auto it = std::find(methods.begin(), methods.end(), "fmm");
  EXPECT_NE(it, methods.end()) << "Expected method 'fmm' was not found in the Python registry.";
}

TEST(ImageProcessingTest, PyExecuteWithValidMethodReturnsImage) {
  // Arrange
  cv::Mat dummyImage = cv::Mat::ones(10, 10, CV_8UC1) * 255;
  std::string validMethod = "fmm";

  // Act
  cv::Mat result = pyImageProcessor->execute(dummyImage, validMethod);

  // Assert
  EXPECT_FALSE(result.empty()) << "Python processor failed to execute valid method: " << validMethod;
}

TEST(ImageProcessingTest, PyExecuteWithInvalidMethodReturnsEmpty) {
  // Arrange
  cv::Mat dummyImage = cv::Mat::ones(10, 10, CV_8UC1) * 255;
  std::string invalidMethod = "this_method_does_not_exist";

  // Act
  cv::Mat result = pyImageProcessor->execute(dummyImage, invalidMethod);

  // Assert
  EXPECT_TRUE(result.empty())
      << "Python processor should catch the error and return an empty matrix for invalid methods.";
}
