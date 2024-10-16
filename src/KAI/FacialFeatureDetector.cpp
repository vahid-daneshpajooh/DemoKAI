#include "FacialFeatureDetector.h"

FacialFeatureDetector::FacialFeatureDetector(const std::string& modelPath) {
    // Load the shape predictor model
    dlib::deserialize(modelPath) >> landmarkPredictor;
}

void FacialFeatureDetector::run(Image& image) {

    // Convert the Image object to dlib image format
    dlib::cv_image<dlib::bgr_pixel> dlibImage = convertToDlibImage(image);

    // Loop through detected face bounding boxes
    std::vector<FacialFeatures> vFeatures;
    for (const auto& [faceBox, conf] : image.getImage_faceBboxes()) {
        
        dlib::rectangle dlibRect(faceBox.x, faceBox.y, faceBox.x + faceBox.width, faceBox.y + faceBox.height);
        
        // Detect facial landmarks using the face bounding box
        dlib::full_object_detection landmarks = landmarkPredictor(dlibImage, dlibRect);

        // extract main facial landmarks (e.g., eye, nose, lips corners)
        FacialFeatures features;
        features.setFFeaturesFromDlib(landmarks);

        vFeatures.push_back(features);
    }
    // Keep Facial Features with image
    image.setFacialFeatures(vFeatures);
}

dlib::cv_image<dlib::bgr_pixel> FacialFeatureDetector::convertToDlibImage(Image& image) {
    cv::Mat imgMat;
    image.getImage_Mat(imgMat);
    return dlib::cv_image<dlib::bgr_pixel>(imgMat);
}