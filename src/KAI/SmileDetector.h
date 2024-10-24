#ifndef SMILEDETECTOR_H
#define SMILEDETECTOR_H

#include "KAITaskInterface.h"
#include "FacialFeatures.h"
#include "Types.h"

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

class SmileDetector: public KAITask {
public:
    SmileDetector(const std::string& modelPath,
                short backendId = 0, short targetId = 0);
    
    void init(const std::map<std::string, Type> params);

    void run(Image& img) override;

private:
    cv::dnn::Net smileNet_;

    //////////////////
    // network params
    //////////////////
    std::string inputName = "input_2";

    cv::Size net_inputSize = cv::Size(224,224); // input size
    float scaleFactor = 1.0f / 255.0f;                // image normalization factor

    bool swapRB = false;
    bool crop = false;
    
    ///
    // helper functions
    ///
};
#endif // SMILEDETECTOR_H