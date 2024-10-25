#ifndef EYEGLASSESDETECTOR_H
#define EYEGLASSESDETECTOR_H

#include "KAITaskInterface.h"
#include "FacialFeatures.h"
#include "Types.h"

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

class EyeglassesDetector: public KAITask {
public:
    EyeglassesDetector(const std::string& modelPath,
                short backendId = 0, short targetId = 0);
    
    void init(const std::map<std::string, Type> params);

    void run(Image& img) override;

private:
    cv::dnn::Net eyeglassesNet_;

    //////////////////
    // network params
    //////////////////
    std::string inputName = "input";
    std::string outputName = "cross_";

    cv::Size net_inputSize = cv::Size(160,160); // input size
    float scaleFactor = 1.0f / 255.0f;                   // image normalization factor

    bool swapRB = false;
    bool crop = false;
    
    ///
    // helper functions
    ///
};
#endif // EYEGLASSESDETECTOR_H