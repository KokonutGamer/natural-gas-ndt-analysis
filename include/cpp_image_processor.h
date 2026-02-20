#ifndef CPP_IMAGE_PROCESSOR_H_
#define CPP_IMAGE_PROCESSOR_H_

#include "include/image_processor.h"

/**
 * TODO document cpp image processor
 */
class CppImageProcessor : public ImageProcessor
{
public:
    /**
     * TODO document execute method
     */
    cv::Mat execute(const cv::Mat &) const override;

    /**
     * TODO document name method
     */
    std::string getName() const override;
};

#endif