#ifndef FACEDETECTOR_H
#define FACEDETECTOR_H

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <string>

class FaceDetector {
public:
    FaceDetector(const std::string& modelPath, const std::string& configPath,
                short backendId = 0, short targetId = 0);
        
    void detectFaces(const cv::Mat& image);

    void setInputName(const std::string name){
        inputName = name;
        return;
    }

    void setOutputName(const std::string name){
        outputName = name;
        return;
    }

    // set model input size
    void setInputSize(const int width, const int height){
        net_inputSize = cv::Size(width, height);
        return;
    }

    void setMeanSubtraction(const std::vector<int> mean){
        imgMean = cv::Scalar(mean[0], mean[1], mean[2]);
        return;
    }

    void setScaleFactor(const double scale){
        scaleFactor = scale;
        return;
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
    cv::Scalar imgMean; // mean subtract
    double scaleFactor = 1.0; // sigma scale

    // cv::dnn::blobFromImage flags
    bool swapRB = false; // swap Red and Blue channels
    bool crop = false; // crop image
    
};

#endif // FACEDETECTOR_H
