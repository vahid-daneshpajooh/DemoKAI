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
    // original image width and height
    auto imgSize = img.getImageSize();

    // resize image to fit model's input size
    cv::Mat img_resized;
    std::vector<float> pad_info = img.resizeImage(img_resized, net_inputSize, true); // padded resize

    cv::Mat detections;
    cv::Mat blob;

    blob = cv::dnn::blobFromImage(img_resized, scaleFactor, net_inputSize, imgMean, swapRB, crop);
    faceNet_.setInput(blob, inputName);
    detections = faceNet_.forward(outputName);

    // post-process network's face detection results
    cv::Mat bboxes(detections.size[2], detections.size[3], CV_32F, detections.ptr<float>());
    auto faces = PostProcess(bboxes, pad_info[0], pad_info[1], pad_info[2], imgSize);

    img.setImage_faceBboxes(faces);
}

std::vector<std::pair<cv::Rect, float>> 
FaceDetector::PostProcess(cv::Mat detections, float pad_w, float pad_h, float scale, const cv::Size& img_size)
{
    auto clip = [](float n, float lower, float upper) {
        return std::max(lower, std::min(n, upper));
    };

    std::vector<std::pair<cv::Rect, float>> faces;
    for(int i = 0; i < detections.rows; i++)
    {
        float confidence = detections.at<float>(i, 2);
    
        if(confidence > conf_thresh)
        {
            // opencv bbox to represent a face
            cv::Rect2i face;

            float x1 = (detections.at<float>(i, 3) * net_inputSize.width - pad_w) / scale;
            float y1 = (detections.at<float>(i, 4) * net_inputSize.height - pad_h) / scale;
            float x2 = (detections.at<float>(i, 5) * net_inputSize.width - pad_w) / scale;
            float y2 = (detections.at<float>(i, 6) * net_inputSize.height - pad_h) / scale;

            // 1. eliminate bboxes that completely fall outside the frame
            if(x2 < 0 || y2 < 0 
              || x1 > img_size.width || y1 > img_size.height){
                continue;
            }

            // 2. ensure at least 90% of bbox's width and height is inside the frame
            float FullDx = x2 - x1;
            float FullDy = y2 - y1;
            
            // TODO: return the face with the highest cof score if all scores < conf_thresh

            //ensures rectangle [tl:(x1, y1), br: (x2, y2)] is inside the frame
            x1 = clip(x1, 0, img_size.width);
            y1 = clip(y1, 0, img_size.height);
            x2 = clip(x2, 0, img_size.width);
            y2 = clip(y2, 0, img_size.height);

            float ClippedDx = x2 - x1;
            float ClippedDy = y2 - y1;
            if  ( (FullDx <= 0.0) || (FullDy <= 0.0)
                || ((ClippedDx/FullDx) <= 0.9) || ((ClippedDy/FullDy) <= 0.9) ) {
                    continue;
            }

            face.x = x1;
            face.y = y1;
            face.width = x2 - x1;
            face.height = y2 - y1;

            faces.push_back(std::pair(face, confidence));
        }
    }

    return faces;
}