#include "include/cpp_image_processor.h"

#include <benchmark/benchmark.h>
#include <filesystem>

class CppImageProcessorFixture : public ::benchmark::Fixture
{
private:
    std::filesystem::path imageDirectory = PROJECT_ROOT_DIR "/images";
protected:
    std::unique_ptr<CppImageProcessor> processor;
    cv::Mat image;

    void SetUp(::benchmark::State &state) override
    {
        processor = std::make_unique<CppImageProcessor>();

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

BENCHMARK_F(CppImageProcessorFixture, CppImageProcessorBenchmark)(benchmark::State &state)
{
    for (auto _ : state)
    {
        processor->execute(image);
    }
}

BENCHMARK_MAIN();