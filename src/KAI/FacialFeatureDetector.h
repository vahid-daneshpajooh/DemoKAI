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
    dlib::shape_predictor landmarkPredictor;

    ///
    // Helper functions
    ///

    // convert OpenCV Image to Dlib Image
    dlib::cv_image<dlib::bgr_pixel> convertToDlibImage(Image& image);
};
#endif // FACIALFEATUREDETECTOR_H