#include "FaceDetector.h"

#include <iostream>

FaceDetector::FaceDetector(const std::string& modelPath,
                const std::string& configPath,
                short backendId, short targetId) {

    faceNet_ = cv::dnn::readNetFromCaffe(configPath, modelPath);
    if (faceNet_.empty()){
        throw std::runtime_error("Error loading face detection model");
    }

    /*{ backend Id    | 0 | Choose one of computation backends:
                        "0: automatically (by default), "
                        "1: Halide language (http://halide-lang.org/), "
                        "2: Intel's Deep Learning Inference Engine (https://software.intel.com/openvino-toolkit), "
                        "3: OpenCV implementation }"
      { target  Id    | 0 | Choose one of target computation devices: "
                        "0: CPU target (by default), "
                        "1: OpenCL, "
                        "2: OpenCL fp16 (half-float precision), "
                        "3: VPU }";
    */
    faceNet_.setPreferableBackend(backendId);
    faceNet_.setPreferableTarget(targetId);
}

void FaceDetector::detectFaces(const cv::Mat& image) {
    cv::Mat detections;
    cv::Mat blob;
    
    // original image width and height
    int imgWidth = image.cols;
    int imgHeight = image.rows;

    // face detection model input size
    net_inputSize = cv::Size(300,300);
    cv::Mat img_resized;
    cv::resize(image, img_resized, net_inputSize);

    imgMean = cv::Scalar(104, 177, 123);

    blob = cv::dnn::blobFromImage(img_resized, scaleFactor, net_inputSize, imgMean, swapRB, crop);
    
    faceNet_.setInput(blob, "data");
    detections = faceNet_.forward("detection_out");

    cv::Mat bboxes(detections.size[2], detections.size[3], CV_32F, detections.ptr<float>());
    
    float conf_thresh = 0.5;
    for(int i = 0; i < bboxes.rows; i++)
    {
        float confidence = bboxes.at<float>(i, 2);
    
        if(confidence > conf_thresh)
        {
            int x1 = static_cast<int>(bboxes.at<float>(i, 3) * imgWidth);
            int y1 = static_cast<int>(bboxes.at<float>(i, 4) * imgHeight);
            int x2 = static_cast<int>(bboxes.at<float>(i, 5) * imgWidth);
            int y2 = static_cast<int>(bboxes.at<float>(i, 6) * imgHeight);

            //ensures rectangle [tl:(x1, y1), br: (x2, y2)] is inside the frame
            x1 = std::max(0, std::min(x1, imgWidth - 1));
            y1 = std::max(0, std::min(y1, imgHeight - 1));
            x2 = std::max(0, std::min(x2, imgWidth - 1));
            y2 = std::max(0, std::min(y2, imgHeight - 1));


            cv::rectangle(image, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(45, 255, 45), 2, 4);
            cv::putText(image, std::to_string(static_cast<int>(confidence*100)),
                        cv::Point(x1+2, y1-5), cv::FONT_HERSHEY_SIMPLEX, 0.5, (45, 45, 255), 2);
        }
    }
}
