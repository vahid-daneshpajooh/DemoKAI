#include "MouthOpenDetector.h"

#include <iostream>

MouthOpenDetector::MouthOpenDetector(const std::string& modelPath,
                                    short backendId, short targetId) {

    // TODO: (thread safety in container envs)
    // cv::dnn::Net is not thread safe, must protect with mutex

    mouthOpenNet_ = cv::dnn::readNetFromTensorflow(modelPath);
    if (mouthOpenNet_.empty()){
        // TODO: catch all runtime errors in KAI Task Manager (?)
        throw std::runtime_error("Mouth Open Detection Task -- Error loading the model.");
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
   
    mouthOpenNet_.setPreferableBackend(backendId);
    mouthOpenNet_.setPreferableTarget(targetId);
}

void MouthOpenDetector::init(const std::map<std::string, Type> params)
{
    // TODO: Error handling
    // e.g., width, height > 0 ; 

    // model's input image size
    int imgWidth, imgHeight;
    if (params.find("NNInputImageWidth") != params.end()
        && params.find("NNInputImageHeight") != params.end()) {
        
        imgWidth = params.at("NNInputImageWidth").get<int>();
        imgHeight = params.at("NNInputImageHeight").get<int>();

        net_inputSize = cv::Size(imgWidth, imgHeight);
    }

    // model's input data normalization (scale factor)
    if (params.find("NNImageNormalization") != params.end()){
        scaleFactor = params.at("NNImageNormalization").get<float>();
    }

    // model's input layer name
    if (params.find("NNInputName") != params.end()) {
        inputName = params.at("NNInputName").get<std::string>();
    }
}

void MouthOpenDetector::run(Image &img)
{
    cv::Mat imgMat;
    img.getImage_Mat(imgMat);
    
    // original image width and height
    int imgWidth = imgMat.cols;
    int imgHeight = imgMat.rows;

    // TODO: update facial features class on Image
    auto vFFeatures = img.getFacialFeatures();
    
    // Loop through detected face bounding boxes
    for (auto& fFeatures : vFFeatures) {

        auto faceBox = fFeatures.getFaceBbox();

        // crop face bbox and resize to model's input size
        cv::Mat faceMat = imgMat(faceBox);
        cv::resize(faceMat, faceMat, net_inputSize);

        cv::Mat blob = cv::dnn::blobFromImage(faceMat, scaleFactor, net_inputSize,
                                              cv::Scalar(0, 0, 0), swapRB, crop);
        
        mouthOpenNet_.setInput(blob, inputName);
        // output: prob. of mouth open
        float probMouthOpen = mouthOpenNet_.forward().at<float>(0,0);

        auto pMouthOpen = std::make_shared<MouthOpen>();
        pMouthOpen->openScore = probMouthOpen;

        // Update Aux Data in Facial Features class
        fFeatures.setAuxData<MouthOpen>(pMouthOpen);
    }

    // TODO: Image class should modify its facial features in-place (instead of clear and copy)
    img.setFacialFeatures(vFFeatures);
}