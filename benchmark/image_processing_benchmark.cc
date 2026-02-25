#include "include/cpp_image_processor.h"
#include "include/py_image_processor.h"

#include <benchmark/benchmark.h>
#include <filesystem>
#include <pybind11/embed.h>

// global image directory path
static const std::filesystem::path imageDirectory = PROJECT_ROOT_DIR "/images";

// processors only ever created once
static std::unique_ptr<CppImageProcessor> cppImageProcessor = std::make_unique<CppImageProcessor>();
static std::unique_ptr<PyImageProcessor> pyImageProcessor = std::make_unique<PyImageProcessor>();

namespace py = pybind11;

class CppImageProcessorFixture : public ::benchmark::Fixture
{
protected:
    cv::Mat image;

    void SetUp(::benchmark::State &state) override
    {
        try
        {
            std::filesystem::path imagePath = imageDirectory / "two_vertical_and_horizontal.tiff";

            image = cv::imread(imagePath, cv::ImreadModes::IMREAD_GRAYSCALE);

            if (image.empty())
            {
                state.SkipWithError("Failed to read from " + imagePath.string());
                return;
            }
        }
        catch (const std::filesystem::filesystem_error &e)
        {
            state.SkipWithError(e.what());
            return;
        }
    }
};

BENCHMARK_F(CppImageProcessorFixture, CppImageProcessorBenchmark)(::benchmark::State &state)
{
    for (auto _ : state)
    {
        cppImageProcessor->execute(image);
    }
}

class PyImageProcessorFixture : public ::benchmark::Fixture
{
protected:
    cv::Mat image;

    void SetUp(::benchmark::State &state) override
    {
        try
        {
            std::filesystem::path imagePath = imageDirectory / "two_vertical_and_horizontal.tiff";

            image = cv::imread(imagePath, cv::ImreadModes::IMREAD_GRAYSCALE);

            if (image.empty())
            {
                state.SkipWithError("Failed to read from " + imagePath.string());
                return;
            }
        }
        catch (const std::filesystem::filesystem_error &e)
        {
            state.SkipWithError(e.what());
            return;
        }
    }
};

BENCHMARK_F(PyImageProcessorFixture, PyImageProcessorBenchmark)(::benchmark::State &state)
{
    for (auto _ : state)
    {
        pyImageProcessor->execute(image);
    }
}

BENCHMARK_MAIN();