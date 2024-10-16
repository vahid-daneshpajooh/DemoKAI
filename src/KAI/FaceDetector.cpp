#include "FaceDetector.h"

#include <iostream>

FaceDetector::FaceDetector(const std::string& modelPath,
                const std::string& configPath,
                short backendId, short targetId) {

    faceNet_ = cv::dnn::readNetFromCaffe(configPath, modelPath);
    if (faceNet_.empty()){
        throw std::runtime_error("Face Detection Task -- Error loading the model.");
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

void FaceDetector::init(const std::map<std::string, Type> params)
{
    // TODO: Error handling
    // e.g., 0 < conf < 1 ; width, height > 0 ; 

    // model's confidence level for the detections
    if (params.find("ConfidenceLevel") != params.end()) {
        conf_thresh = params.at("ConfidenceLevel").get<float>();
    }

    // model's input image size
    int imgWidth, imgHeight;
    if (params.find("NNInputImageWidth") != params.end()
        && params.find("NNInputImageHeight") != params.end()) {
        
        imgWidth = params.at("NNInputImageWidth").get<int>();
        imgHeight = params.at("NNInputImageHeight").get<int>();

        net_inputSize = cv::Size(imgWidth, imgHeight);
    }

    // model's input pre-processing
    // 1. mean subtraction
    if (params.find("NNMeanSubtraction") != params.end()) {
        std::vector<float> meanVec = params.at("NNMeanSubtraction").get<std::vector<float>>();
        
        imgMean[0] = meanVec[0];
        imgMean[1] = meanVec[1];
        imgMean[2] = meanVec[2];
    }

    // model's input/output layer names
    if (params.find("NNInputName") != params.end()) {
        inputName = params.at("NNInputName").get<std::string>();
    }
    if (params.find("NNOutputName") != params.end()) {
        outputName = params.at("NNOutputName").get<std::string>();
    }
}

void FaceDetector::run(Image &img)
{
    cv::Mat imgMat;
    img.getImage_Mat(imgMat);
    
    // original image width and height
    int imgWidth = imgMat.cols;
    int imgHeight = imgMat.rows;

    cv::Mat img_resized;
    cv::resize(imgMat, img_resized, net_inputSize);

    
    cv::Mat detections;
    cv::Mat blob;

    blob = cv::dnn::blobFromImage(img_resized, scaleFactor, net_inputSize, imgMean, swapRB, crop);
    faceNet_.setInput(blob, inputName);
    detections = faceNet_.forward(outputName);

    cv::Mat bboxes(detections.size[2], detections.size[3], CV_32F, detections.ptr<float>());
    
    std::vector<std::pair<cv::Rect, float>> faces;
    for(int i = 0; i < bboxes.rows; i++)
    {
        float confidence = bboxes.at<float>(i, 2);
    
        if(confidence > conf_thresh)
        {
            // opencv bbox to represent a face
            cv::Rect2i face;

            int x1 = static_cast<int>(bboxes.at<float>(i, 3) * imgWidth);
            int y1 = static_cast<int>(bboxes.at<float>(i, 4) * imgHeight);
            int x2 = static_cast<int>(bboxes.at<float>(i, 5) * imgWidth);
            int y2 = static_cast<int>(bboxes.at<float>(i, 6) * imgHeight);

            //ensures rectangle [tl:(x1, y1), br: (x2, y2)] is inside the frame
            x1 = std::max(0, std::min(x1, imgWidth - 1));
            y1 = std::max(0, std::min(y1, imgHeight - 1));
            x2 = std::max(0, std::min(x2, imgWidth - 1));
            y2 = std::max(0, std::min(y2, imgHeight - 1));

            face.x = x1;
            face.y = y1;
            face.width = x2 - x1;
            face.height = y2 - y1;

            faces.push_back(std::pair(face, confidence));
        }
    }
    img.setImage_faceBboxes(faces);
}
