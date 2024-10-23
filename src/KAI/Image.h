#ifndef IMAGE_H
#define IMAGE_H

#include <vector>
#include <mutex>

#include <opencv2/opencv.hpp>
#include "FacialFeatures.h"

class Image {
public:
    Image(const std::string& img_path){

        // we use linux standard file separator "/"
        size_t pos = img_path.find_last_of('/');
        if(pos == std::string::npos){
            // No directory separator found, return the entire path
            imageName = img_path;
        }
        else {
            // everything (inlcuding extension) after last '/'
            imageName = img_path.substr(pos + 1);
        }

        imgMat = cv::imread(img_path);
    }

    void getImage_Mat(cv::Mat& outMat){
        
        std::lock_guard<std::mutex> lock(imageMutex); // protect access
        
        imgMat.copyTo(outMat);
    }

    std::vector<std::pair<cv::Rect, float>> getImage_faceBboxes(){
        
        std::lock_guard<std::mutex> lock(imageMutex); // protect access
        
        return faceBboxes;
    }

    void setImage_faceBboxes(const std::vector<std::pair<cv::Rect, float>>& bboxes){
        
        std::lock_guard<std::mutex> lock(imageMutex); // protect access

        // clear current vector elements
        faceBboxes.clear();
        faceBboxes = bboxes;
    }

    void setFacialFeatures(std::vector<FacialFeatures> features){
        
        std::lock_guard<std::mutex> lock(imageMutex); // protect access

        // clear current vector elements
        vFacialFeatures.clear();
        vFacialFeatures = features;
    }

    std::vector<FacialFeatures> getFacialFeatures(){
        
        std::lock_guard<std::mutex> lock(imageMutex); // protect access
        
        return vFacialFeatures;
    }

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
        overlayFaceFeaturesOnImage(outMat);

        // Print estimated roll, yaw, and pitch degrees
        overlayFacePoseOnImage(outMat);
    }

    std::string getName(){
        return imageName;
    }

private:
    
    // Mutex to protect shared image data
    std::mutex imageMutex;

    std::string imageName; // image file name
    cv::Mat imgMat;        // image cv::Mat variable

    // Face Detection results
    // bounding boxes (x, y, width, height) and confidence scores
    std::vector<std::pair<cv::Rect, float>> faceBboxes;

    // Facial Features (landmarks, smile, gaze, eyeglasses, etc)
    // (for all detected faces)
    std::vector<FacialFeatures> vFacialFeatures;
    
    //////////////////////////////////
    // visualization utility functions
    //////////////////////////////////

    // Draw rectangle around detected faces
    void overlayFaceOnImage(cv::Mat& overlayImg){
        for(const auto& faceBbox: faceBboxes){
            cv::rectangle(overlayImg, faceBbox.first, cv::Scalar(45, 255, 45), 2, 4);
            
            cv::Point topRight(faceBbox.first.x + faceBbox.first.width - 25, faceBbox.first.y-5);
            cv::putText(overlayImg, std::to_string(static_cast<int>(faceBbox.second*100)),
                        topRight, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 45, 45), 2);
        }
    }

    // Draw facial landmarks on detected faces
    void overlayFaceLandmarkOnImage(cv::Mat& overlayImg){
        
        // get FacialFeatures class for each face detected in image
        for(const auto& faceFeatures: vFacialFeatures){
            // get facial landmarks (e.g., eye, lip, nose corners)
            auto landmarks = faceFeatures.getFacialLandmarks();

            // draw a circle to show each landmark on the face
            for (const auto& landmark: landmarks){
                // cv::Scalar(45, 255, 45) -> green
                cv::circle(overlayImg, cv::Point(landmark.mX, landmark.mY), 2, cv::Scalar(45, 255, 45), 2);
            }
        }
    }

    // Draw ALL detected facial feature points
    void overlayFaceFeaturesOnImage(cv::Mat& overlayImg){
        
        // get FacialFeatures class for each face detected in image
        for(const auto& faceFeatures: vFacialFeatures){
            // get facial features (e.g., Dlib 68 points)
            auto featurePoints = faceFeatures.getFacialFeatures();

            // draw a circle to show each feature point on the face
            for (const auto& point: featurePoints){
                // cv::Scalar(255, 45, 45) -> blue
                cv::circle(overlayImg, point, 1, cv::Scalar(255, 45, 45), 1);
            }
        }
    }

    void overlayFacePoseOnImage(cv::Mat overlayImg){

        // get FacialFeatures class for each face detected in image
        for(const auto& faceFeatures: vFacialFeatures) {
            // get face pose vector
            auto facePose = faceFeatures.getFacePose();
            std::string text = "Roll:" + std::to_string(static_cast<int>(facePose[0])) + 
                              ", Yaw:" + std::to_string(static_cast<int>(facePose[1])) + 
                              ", Pitch:" + std::to_string(static_cast<int>(facePose[2]));

            auto faceBox = faceFeatures.getFaceBbox(); // only outputs bbox (no conf)
            cv::putText(overlayImg, text, cv::Point(faceBox.x+2, faceBox.y-5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 45, 45), 1);
        }
    }
};
#endif // IMAGE_H