#include "include/cpp_image_processor.h"
#include "include/py_image_processor.h"

#include <benchmark/benchmark.h>
#include <filesystem>
#include <pybind11/embed.h>

// global image directory path
static const std::filesystem::path imageDirectory = PROJECT_ROOT_DIR "/images";
static cv::Mat image;

// processors only ever created once
static std::unique_ptr<CppImageProcessor> cppImageProcessor = std::make_unique<CppImageProcessor>();
static std::unique_ptr<PyImageProcessor> pyImageProcessor = std::make_unique<PyImageProcessor>();

auto peformanceBenchmark = [](::benchmark::State &state, const ImageProcessor * processor, const cv::Mat &image, const std::string method)
{
    for (auto _ : state)
    {
        processor->execute(image, method);
    }
};

int main(int argc, char *argv[])
{
    // set up the static image for processing
    cv::Mat image;
    try
    {
        std::filesystem::path imagePath = imageDirectory / "two_vertical_and_horizontal.tiff";

        image = cv::imread(imagePath, cv::ImreadModes::IMREAD_GRAYSCALE);

        if (image.empty())
        {
            std::cerr << "Failed to read from " << imagePath.string() << std::endl;
            return EXIT_FAILURE;
        }
    }
    catch (const std::filesystem::filesystem_error &e)
    {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    // register single cpp method for processing
    ::benchmark::RegisterBenchmark(cppImageProcessor->getName(""), peformanceBenchmark, cppImageProcessor.get(), image, "fmm");

    // register all python methods for processing
    for (const auto &method : pyImageProcessor->getMethods())
    {
        ::benchmark::RegisterBenchmark(pyImageProcessor->getName(method), peformanceBenchmark, pyImageProcessor.get(), image, method);
    }

    ::benchmark::Initialize(&argc, argv);

    if (::benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;

    ::benchmark::RunSpecifiedBenchmarks();
}