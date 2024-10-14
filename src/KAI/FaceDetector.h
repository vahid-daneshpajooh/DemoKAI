#ifndef FACEDETECTOR_H
#define FACEDETECTOR_H

#include "KAITaskInterface.h"
#include "Types.h"

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <string>

class FaceDetector: public KAITask {
public:
    FaceDetector(const std::string& modelPath, const std::string& configPath,
                short backendId = 0, short targetId = 0);

    void init(const std::map<std::string, Type> params);

    void run(Image& img) override;
    
    void detectFaces(Image& image);

    void setInputName(const std::string name){
        inputName = name;
    }

    void setOutputName(const std::string name){
        outputName = name;
    }

    // set model input size
    void setInputSize(const int width, const int height){
        if (width > 0 && height > 0)
            net_inputSize = cv::Size(width, height);
    }

    void setConfidenceLevel(const float thresh){
        if(thresh > 0 && thresh <= 1)
            conf_thresh = thresh;
    }

    void setMeanSubtraction(const std::vector<float> mean){
        imgMean = cv::Scalar(mean[0], mean[1], mean[2]);
    }

    void setScaleFactor(const float scale){
        scaleFactor = scale;
    }

private:

    cv::dnn::Net faceNet_;

    //////////////////
    // network params
    //////////////////

    // input/output names
    std::string outputName = "detection_out";
    std::string inputName = "data";
    
    // input size
    cv::Size net_inputSize = cv::Size(300,300);
    
    // normalize image
    // x_n = (x - mean) / sigma
    cv::Scalar imgMean = (104.0, 177.0, 123.0);
; // mean subtract
    float scaleFactor = 1.0; // sigma scale

    // cv::dnn::blobFromImage flags
    bool swapRB = false; // swap Red and Blue channels
    bool crop = false; // crop image
    
    // confidence threshold
    float conf_thresh = 0.5;
};

#endif // FACEDETECTOR_H
