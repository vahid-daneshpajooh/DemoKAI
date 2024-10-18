#ifndef FACIALFEATUREDETECTOR_H
#define FACIALFEATUREDETECTOR_H

#include "KAITaskInterface.h"
#include "FacialFeatures.h"

#include <dlib/image_processing.h>
#include <dlib/image_io.h>
#include <dlib/opencv.h> // operate OpenCV and Dlib

#include <string>

class FacialFeatureDetector: public KAITask {
public:
    FacialFeatureDetector(const std::string& modelPath);
    
    // Override the run function to detect facial landmarks
    void run(Image& image) override;
    
private:
    
    /////////////////
    // network params
    //////////////////

    // input size
    cv::Size net_inputSize = cv::Size(500, 500);

    // Dlib inference head
    dlib::shape_predictor landmarkPredictor;


    ///
    // Helper functions
    ///

    // scale Dlib model's predictions to actual image size
    void adjustLandmarksScale(dlib::full_object_detection& landmarks, float scale, const cv::Size& out_size);

    // resize image and maintain aspect ratio
    float resizeImage(const cv::Mat& src, cv::Mat& dst, const cv::Size& out_size);

    // scale face bbox coordinates
    void scaleCoordinates(cv::Point& point, float scale, bool pyrUp, const cv::Size& img_size);
};
#endif // FACIALFEATUREDETECTOR_H