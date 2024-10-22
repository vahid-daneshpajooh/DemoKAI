#include "FacialFeatureDetector.h"

#include <algorithm>

FacialFeatureDetector::FacialFeatureDetector(const std::string& modelPath) {
    // Load the shape predictor model
    dlib::deserialize(modelPath) >> landmarkPredictor;
}

void FacialFeatureDetector::run(Image& image) {

    cv::Mat origImg;
    image.getImage_Mat(origImg);

    // resize image to Dlib model's input size
    // [Dlib Bug]: Dlib model doesn't require specific size,
    //       but the dlib::shape_predictor() function
    //       throws SegFault for large images; e.g. 4236 x 3648
    
    cv::Mat img_resized;
    float scale = resizeImage(origImg, img_resized, net_inputSize);

    // Convert cv image to dlib image format
    dlib::cv_image<dlib::bgr_pixel> dlibImage(img_resized);
    
    // Loop through detected face bounding boxes
    std::vector<FacialFeatures> vFeatures;
    for (const auto& [faceBox, conf] : image.getImage_faceBboxes()) {
        
        // adjust face box scale to match Dlib model's input size
        // pyrUp = false (scaling from large input image to smaller net input)
        cv::Point tlPoint = faceBox.tl();
        cv::Point brPoint = faceBox.br();
        scaleCoordinates(tlPoint, scale, false, img_resized.size());
        scaleCoordinates(brPoint, scale, false, img_resized.size());

        dlib::rectangle dlibRect(tlPoint.x, tlPoint.y, brPoint.x, brPoint.y);
        // detect facial landmarks using the "color image"
        dlib::full_object_detection landmarks = landmarkPredictor(dlibImage, dlibRect);
        
        // scale landmarks back to original image size
        adjustLandmarksScale(landmarks, scale, origImg.size());

        // extract main facial landmarks (e.g., eye, nose, lips corners)
        FacialFeatures features;
        features.setFaceBbox(std::make_pair(faceBox, conf));
        features.setFFeaturesFromDlib(landmarks);

        vFeatures.push_back(features);
    }
    // Keep Facial Features with image
    image.setFacialFeatures(vFeatures);
}

void FacialFeatureDetector::adjustLandmarksScale(dlib::full_object_detection &landmarks,
                                                float scale, const cv::Size& out_size)
{
    for (int i = 0; i < landmarks.num_parts(); ++i) {
        cv::Point fPoint;
        fPoint.x = static_cast<int>(landmarks.part(i).x());
        fPoint.y = static_cast<int>(landmarks.part(i).y());

        scaleCoordinates(fPoint, scale, true, out_size);

        landmarks.part(i) = dlib::point(fPoint.x, fPoint.y);
    }
}

float FacialFeatureDetector::resizeImage(const cv::Mat& src,
                            cv::Mat& dst, const cv::Size& out_size){
    // input:  1. original image
    //         2. desired output size
    //
    // output: resized image with same aspect ratio as input

    // source and dest. image dimensions
    // dest = Dlib model's input size (e.g., 500x500)
    auto in_h = static_cast<float>(src.rows);
    auto in_w = static_cast<float>(src.cols);
    float out_h = out_size.height;
    float out_w = out_size.width;

    // scale factor equal to the min of  [w_in/w_out] or [h_in/h_out]
    float scale = std::min(out_w / in_w, out_h / in_h);

    // resize image (we maintain aspect ratio)
    int mid_h = static_cast<int>(in_h * scale);
    int mid_w = static_cast<int>(in_w * scale);
    cv::resize(src, dst, cv::Size(mid_w, mid_h));

    return scale;
}

void FacialFeatureDetector::scaleCoordinates(cv::Point& point, float scale,
                                            bool pyrUp, const cv::Size& img_size) {
    
    // pyrUp = true: scaling coordinates up to original image size
    // pyrUp = false: scaling coordinates down from original image size to model's input size

    // helper function to clip scaled boxes to image dims
    auto clip = [](float n, float lower, float upper) {
        return std::max(lower, std::min(n, upper));
    };

    float x_new, y_new;
    if(pyrUp) {
        x_new = point.x / scale;  // x scaling
        y_new = point.y / scale;  // y scaling
    }
    else {
        x_new = point.x * scale;  // x scaling
        y_new = point.y * scale;  // y scaling
    }
    

    // clip scaled coordinates if necessary
    x_new = clip(x_new, 0, img_size.width);
    y_new = clip(y_new, 0, img_size.height);

    point = cv::Point(x_new, y_new);
}