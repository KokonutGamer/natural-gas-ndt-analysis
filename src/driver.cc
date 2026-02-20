#include "include/cpp_image_processor.h"
#include "include/cxxopts.hpp"
#include "include/image_processor.h"
#include "include/py_image_processor.h"

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

int main(int argc, char *argv[])
{
    cxxopts::Options options("NDT Image Analyzer", "Non-destructive testing "
                                                   "image processing software developed for detecting microcracks and "
                                                   "micro pits on the Raspberry Pi 5");

    options.add_options()("p,python", "Run with the embedded Python interpreter")("h,help", "Print usage")("i,image", "Process an image", cxxopts::value<std::string>()->default_value("./images/two_vertical_and_horizontal.tiff"))("o,output", "Output the processed image to a file", cxxopts::value<std::string>()->default_value("./processed/p_two_vertical_and_horizontal.tiff"));

    auto result = options.parse(argc, argv);

    // return early on help
    if (result.count("help"))
    {
        std::cout << options.help() << std::endl;
        return EXIT_SUCCESS;
    }

    // ImageProcessor *processor = nullptr;
    std::unique_ptr<ImageProcessor> processor;
    if (result["python"].as<bool>())
    {
        processor = std::make_unique<PyImageProcessor>();
    }
    else
    {
        processor = std::make_unique<CppImageProcessor>();
    }

    // try accessing the input image first
    try
    {
        // sanitize the image path and check if it exists
        std::filesystem::path imagePath = std::filesystem::canonical(result["image"].as<std::string>());

        // read from the image
        cv::Mat image = cv::imread(imagePath, cv::ImreadModes::IMREAD_GRAYSCALE);

        // check if read was successful
        if (image.empty())
        {
            std::cerr << "Failed to read from " << result["image"].as<std::string>() << std::endl;
            return EXIT_FAILURE;
        }

        // process the image using the specified processor
        cv::Mat processedImage = processor->execute(image);

        // write the image to a new file
        cv::imwrite(result["output"].as<std::string>(), processedImage);
    }
    catch (const std::filesystem::filesystem_error &e)
    {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << processor->getName() << " processed " << result["image"].as<std::string>() << " to " << result["output"].as<std::string>() << std::endl;
    return EXIT_SUCCESS;
}