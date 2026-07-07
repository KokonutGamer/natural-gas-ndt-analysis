#include "include/cpp_image_processor.h"
#include "include/image_processor.h"
#include "include/py_image_processor.h"

#include <gtest/gtest.h>
#include <pybind11/embed.h>

// processors only ever created once
static std::unique_ptr<CppImageProcessor> cppImageProcessor;
static std::unique_ptr<PyImageProcessor> pyImageProcessor;

// ==========================================
// C++ Processor Tests
// ==========================================

TEST(NameTest, CppImageProcessorHasCorrectName) {
  // Act
  std::string name = cppImageProcessor->getName("fmm");

  // Assert
  EXPECT_EQ(name, "C++ image processor");
}

// ==========================================
// Python Processor Tests
// ==========================================

TEST(NameTest, PyImageProcessorHasCorrectName) {
  // Act
  std::string name = pyImageProcessor->getName("fmm");

  // Assert
  EXPECT_EQ(name, "Filter-Mask-Morph (FMM) Python image processor");
}

int main(int argc, char *argv[]) {
  // initialize processors inside main
  cppImageProcessor = std::make_unique<CppImageProcessor>();
  pyImageProcessor = std::make_unique<PyImageProcessor>();

  ::testing::InitGoogleTest(&argc, argv);
  int result = RUN_ALL_TESTS();

  return result;
}