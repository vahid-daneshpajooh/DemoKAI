#ifndef TFLITEFACIALFEATUREDETECTOR_H
#define TFLITEFACIALFEATUREDETECTOR_H

#include "tensorflow/lite/core/interpreter_builder.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/optional_debug_tools.h"

#include "KAITaskInterface.h"
#include "FacialFeatures.h"
#include "Types.h"

#include <string>

class TFLiteFacialFeatureDetector: public KAITask {
public:
    /**
     * @brief Construct a new TFLiteFacialFeatureDetector object
     * 
     * @param modelPath 
     */
    TFLiteFacialFeatureDetector(const std::string& modelPath);
    
    /**
     * @brief 
     * 
     * @param params 
     */
    void init(const std::map<std::string, Type> params);

    /**
     * @brief Override the run function to detect facial landmarks
     * 
     * @param image 
     */
    void run(Image& image) override;
    
private:
    
    /////////////////
    // network params
    //////////////////
    
    // Flatbuffer model loader
    std::unique_ptr<tflite::FlatBufferModel> faceMeshNet_;
    
    // Note: all Interpreters should be built with the InterpreterBuilder,
    // which allocates memory for the Interpreter and does various set up
    // tasks so that the Interpreter can read the provided model.
    std::unique_ptr<tflite::Interpreter> interpreter;

    float* faceMeshNet_inputLayer;
    float* faceMeshNet_outputLayer;

    // input size
    cv::Size net_inputSize = cv::Size(192, 192);
    
    // num facial landmarks generated by TFLite model
    const int TFLite_numFaceLandmarks = 468;

    // margin value (in percentage) to increase face box on each side
    float margin = 0.25;

    ///////////////////
    // Helper functions
    ///////////////////
    
    /**
     * @brief Increase face box margin on each side
     * 
     * @param faceBox  - face bounding box
     * @param img_size - image size (width, height)
     * @param margin   - margin factor in range [0, 1] to increase face box on each side
     * @return cv::Rect  square (width = height) face bounding box
     */
    cv::Rect increaseFaceMargin(const cv::Rect& faceBox, cv::Size img_size, float margin);

    // scale face bbox coordinates
    void scaleCoordinates(cv::Point& point, float pad_w, float pad_h, float scale, const cv::Size& img_size);

    // get face landmarks from TFlite output layer
    cv::Point getFaceLandmarkAt(int index, float* tensor);
};
#endif // TFLITEFACIALFEATUREDETECTOR_H