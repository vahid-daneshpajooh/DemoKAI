#ifndef IMAGE_H
#define IMAGE_H

#include <vector>
#include <opencv2/opencv.hpp>
#include "FacialFeatures.h"

class Image {
public:
    Image(const std::string& img_path){
        imgMat = cv::imread(img_path);
    }

    void getImage_Mat(cv::Mat& outMat){
        imgMat.copyTo(outMat);
    }

    std::vector<std::pair<cv::Rect, float>> getImage_faceBboxes(){
        return faceBboxes;
    }

    void setImage_faceBboxes(const std::vector<std::pair<cv::Rect, float>>& bboxes){
        // clear current vector elements
        faceBboxes.clear();
        faceBboxes = bboxes;
    }

    void setFacialFeatures(FacialFeatures features){
        facialFeatures = features;
    }

    /*
    void addFacialLandmarks(const FFeatureLocation& ffLocs){

    }
    */

    void getImage_faceOn(cv::Mat& outMat){
        if (imgMat.empty()){
            return;
        }

        if(outMat.empty()){
            outMat = imgMat.clone();
        }

        overlayFaceOnImage(outMat);
    }

    void getImage_faceFeaturesOn(cv::Mat& outMat){
        if (imgMat.empty()){
            return;
        }

        if(outMat.empty()){
            outMat = imgMat.clone();
        }
        
        overlayFaceLandmarkOnImage(outMat);
    }

private:
    cv::Mat imgMat;

    // Face Detection results
    // bounding boxes (x, y, width, height) and confidence scores
    std::vector<std::pair<cv::Rect, float>> faceBboxes;

    // Facial Features (landmarks, smile, gaze, eyeglasses, etc)
    FacialFeatures facialFeatures;

    // visualization utility functions

    // Draw rectangle around detected faces
    void overlayFaceOnImage(cv::Mat& overlayImg){
        for(const auto& faceBox: faceBboxes){
            cv::rectangle(overlayImg, faceBox.first, cv::Scalar(45, 255, 45), 2, 4);
            cv::putText(overlayImg, std::to_string(static_cast<int>(faceBox.second*100)),
                        cv::Point(faceBox.first.x+2, faceBox.first.y-5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 45, 45), 2);
        }
    }

    // Draw facial landmarks on detected faces
    void overlayFaceLandmarkOnImage(cv::Mat& overlayImg){
        
        auto vFFeatures = facialFeatures.getFacialLandmarks();
        for (const auto& feature: vFFeatures){
            cv::circle(overlayImg, cv::Point(feature.mX, feature.mY), 2, cv::Scalar(45, 255, 45), 2);
        }
    }
};
#endif // IMAGE_H