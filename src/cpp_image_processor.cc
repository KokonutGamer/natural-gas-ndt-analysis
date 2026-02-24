#include "include/cpp_image_processor.h"

cv::Mat CppImageProcessor::execute(const cv::Mat &image) const
{
    // copy constructor
    cv::Mat processedImage = image;

    // filter parameters
    int numberOfFilters = 4;
    int kernelSize = 3;
    cv::Mat maxKernel = cv::getStructuringElement(
        cv::MORPH_RECT,
        cv::Size(kernelSize, kernelSize),
        cv::Point(kernelSize / 2, kernelSize / 2));

    for (int i = 0; i < numberOfFilters; i++)
    {
        // max filter
        cv::dilate(processedImage, processedImage, maxKernel);

        // median filter
        cv::medianBlur(processedImage, processedImage, kernelSize);
    }

    // find the threshold based on the bottom 5 percentile
    int histogram[256] = {0};

    // increment frequencies of each pixel value
    for (int r = 0; r < processedImage.rows; r++)
    {
        for (int c = 0; c < processedImage.cols; c++)
        {
            histogram[processedImage.ptr<uchar>(r)[c]]++;
        }
    }

    double thresholdValue{0};
    size_t numberInThreshold = static_cast<size_t>(processedImage.rows * processedImage.cols * 0.05);

    // find the threshold value by traversing the histogram
    for (int i = 0; i < 256; i++)
    {
        if (histogram[i] <= numberInThreshold)
        {
            numberInThreshold -= histogram[i];
            continue;
        }
        // threshold value found
        thresholdValue = static_cast<double>(i);
        break;
    }

    // apply a binary mask on the image
    cv::threshold(
        processedImage,
        processedImage,
        thresholdValue,
        255.0,
        cv::ThresholdTypes::THRESH_BINARY_INV);

    // morphological operation parameters
    int numberOfMorphs = 4;
    kernelSize = 5;
    cv::Mat morphKernel = cv::getStructuringElement(
        cv::MORPH_RECT,
        cv::Size(kernelSize, kernelSize),
        cv::Point(kernelSize / 2, kernelSize / 2));

    for (int i = 0; i < numberOfMorphs; i++)
    {
        // open
        cv::morphologyEx(
            processedImage,
            processedImage,
            cv::MorphTypes::MORPH_OPEN,
            morphKernel);

        // close
        cv::morphologyEx(
            processedImage,
            processedImage,
            cv::MorphTypes::MORPH_CLOSE,
            morphKernel);
    }

    return processedImage;
}

std::string CppImageProcessor::getName() const
{
    return "C++ image processor";
}
