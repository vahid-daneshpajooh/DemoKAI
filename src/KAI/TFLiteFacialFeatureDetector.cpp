#include "TFLiteFacialFeatureDetector.h"

#include <iostream>

// helper function to clip scaled boxes to image dims
auto clip = [](float n, float lower, float upper) {
    return std::max(lower, std::min(n, upper));
};

TFLiteFacialFeatureDetector::TFLiteFacialFeatureDetector(const std::string& modelPath) {
    
    // Load the TFLite model
    faceMeshNet_ = tflite::FlatBufferModel::BuildFromFile(modelPath.c_str());
    if(faceMeshNet_ == nullptr){
        throw std::runtime_error("Facial Features Detection Task"
                "-- Error loading the TFLite model.");
    }
    
    // Build the interpreter with the InterpreterBuilder.
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*faceMeshNet_.get(), resolver);
    builder(&interpreter);
    if (interpreter == nullptr){
        throw std::runtime_error("Facial Features Detection Task"
                "-- Failed to build the TFLite model interpreter.");
    }

    // Allocate tensor buffers.
    if (interpreter->AllocateTensors() != kTfLiteOk){
        throw std::runtime_error("Facial Features Detection Task"
                "-- Failed to allocate tensors for the TFLite model interpreter.");
    }
    
    // TODO: set num threads for the interpreter
    // interpreter->SetNumThreads(numThreads);
}

void TFLiteFacialFeatureDetector::init(const std::map<std::string, Type> params){
    
    // model's input image size
    int imgWidth, imgHeight;
    if (params.find("NNInputImageWidth") != params.end()
        && params.find("NNInputImageHeight") != params.end()) {
        
        imgWidth = params.at("NNInputImageWidth").get<int>();
        imgHeight = params.at("NNInputImageHeight").get<int>();

        net_inputSize = cv::Size(imgWidth, imgHeight);
    }
}

void TFLiteFacialFeatureDetector::run(Image& image) {

    // original image width and height
    auto imgSize = image.getImageSize();
    
    std::vector<FacialFeatures> vFeatures;
    for (const auto& [faceBox, conf] : image.getImage_faceBboxes()) {
        
        // increase face box margin by 25% on each side
        auto newFaceBox = increaseFaceMargin(faceBox, imgSize, margin);

        /// preprocess image
        // 1. resize face image to fit model's input size (e.g., 192x192)
        cv::Mat faceMat;
        std::vector<float> pad_info = image.resizeImage(newFaceBox, faceMat, net_inputSize, true); // padded resize

        // 2. rearrange channels to RGB
        cv::cvtColor(faceMat, faceMat, cv::COLOR_BGR2RGB);

        // 3. convert CV_8UC3 pixel values to CV_32FC3
        //    normalized between (-1, 1)
        float inputNorm_mean =  127.5f;
        float inputNorm_std  =  127.5f;
        faceMat.convertTo(faceMat, CV_32FC3, 1 / inputNorm_std, - inputNorm_mean / inputNorm_std);

        // 4. Fill input/output buffers
        faceMeshNet_inputLayer = interpreter->typed_input_tensor<float>(0);
        faceMeshNet_outputLayer = interpreter->typed_output_tensor<float>(0);
        std::memcpy(faceMeshNet_inputLayer, faceMat.data, faceMat.total() * faceMat.elemSize());

        /// Run inference
        if (interpreter->Invoke() != kTfLiteOk){
            throw std::runtime_error("Facial Features Detection Task"
                    "-- Failed to invoke the TFLite model interpreter."); 
        }

        /// Post process
        // Read output buffers
        std::vector<cv::Point> landmarks;
        for (int i = 0; i < TFLite_numFaceLandmarks; ++i) {
            
            // get face landmark i from output layer 
            cv::Point point = getFaceLandmarkAt(i, faceMeshNet_outputLayer);
            
            // scale feature points to original image
            scaleCoordinates(point, pad_info[0], pad_info[1], pad_info[2], newFaceBox.size());
            
            landmarks.push_back(point + newFaceBox.tl());
        }

        // extract main facial landmarks (e.g., eye, nose, lips corners)
        FacialFeatures features;
        features.setFaceBbox(std::make_pair(faceBox, conf));
        features.setFFeaturesFromTFLite(landmarks);

        vFeatures.push_back(features);
    }
    // Keep Facial Features with image
    image.setFacialFeatures(vFeatures);
}

cv::Rect TFLiteFacialFeatureDetector::increaseFaceMargin(const cv::Rect& faceBox, 
                                                        cv::Size imgSize, float margin) {

    // box's top-left and bottom-right coordinates
    cv::Point tlPoint = faceBox.tl();
    cv::Point brPoint = faceBox.br();

    // face box width and height
    float w = static_cast<float>(faceBox.width);
    float h = static_cast<float>(faceBox.height);

    // face box aspect ratio: w/h
    float ar = w/h;

    // increase box dims so that the min. dim. is increased by (2*margin)%,
    // and new dims are equal w_new = h_new
    float w_new, h_new;
    if(ar < 1) {
        w_new =          w * (1 + 2*margin);
        h_new =   ar   * h * (1 + 2*margin);
    }
    else {
        w_new = (1/ar) * w * (1 + 2*margin);
        h_new =          h * (1 + 2*margin);
    }

    // increase box margin on each side using w_new and h_new
    int margin_h = (h_new - h) / 2;
    int margin_w = (w_new - w) / 2;

    // Also, ensure new bbox stays inside the frame
    int x1_new = clip(tlPoint.x - margin_w, 0, imgSize.width);
    int y1_new = clip(tlPoint.y - margin_h, 0, imgSize.height);
    int x2_new = clip(brPoint.x + margin_w, 0, imgSize.width);
    int y2_new = clip(brPoint.y + margin_h, 0, imgSize.height);
    
    // convert to bbox [x, y, w, h] and return
    return cv::Rect(x1_new, y1_new, (x2_new - x1_new), (y2_new - y1_new));
}

void TFLiteFacialFeatureDetector::scaleCoordinates(cv::Point& point, float pad_w, 
                                float pad_h, float scale, const cv::Size& img_size) {

    float x_new, y_new;
    x_new = (point.x - pad_w) / scale;  // x scaling
    y_new = (point.y - pad_h) / scale;  // y scaling
    

    // clip scaled coordinates if necessary
    x_new = clip(x_new, 0, img_size.width);
    y_new = clip(y_new, 0, img_size.height);

    point = cv::Point(x_new, y_new);
}

cv::Point TFLiteFacialFeatureDetector::getFaceLandmarkAt(int index, float* tensor){
    
    float _x = tensor[index * 3 + 0];
    float _y = tensor[index * 3 + 1];
    float _z = tensor[index * 3 + 2]; // depth info.

    return cv::Point(_x,_y);
}