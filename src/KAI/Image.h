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

    cv::Size getImageSize(){
        return imgMat.size();
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

    //////////////////////////////////
    // Image manipulation functions
    //////////////////////////////////
    //@{
    /**
     * @brief (Padded) resize
     * @note  maintains the image aspect ratio.
     *        Further, pads the image to out_size if instructed (pad = true).
     * @param box - an roi in input image
     * @note  when <box> is provided, the transformation is applied on roi = img(box).
     * @param dst - output image
     * @param out_size - desired output size
     * @param pad - pads image to ensure output size
     * @return padding information - pad width, pad height and zoom scale
     * @note   pad width and height = -1, when param pad = false.
     */
    std::vector<float> resizeImage(cv::Mat& dst, const cv::Size& out_size = cv::Size(300, 300),
                                    bool pad = false) {

        cv::Mat src;
        imgMat.copyTo(src);

        // source and dest. image dimensions
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

        // pad image to fit to out_size
        int top = -1, down = -1, left = -1, right = -1;
        if(pad){
            top = (static_cast<int>(out_h) - mid_h) / 2;
            down = (static_cast<int>(out_h)- mid_h + 1) / 2;
            left = (static_cast<int>(out_w)- mid_w) / 2;
            right = (static_cast<int>(out_w)- mid_w + 1) / 2;

            cv::copyMakeBorder(dst, dst, top, down, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
        }

        std::vector<float> pad_info{static_cast<float>(left), static_cast<float>(top), scale};
        return pad_info;
    }

    std::vector<float> resizeImage(cv::Rect box, cv::Mat& dst, const cv::Size& out_size = cv::Size(300, 300),
                                    bool pad = false) {

        // image roi
        cv::Mat src = imgMat(box);

        // source and dest. image dimensions
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

        // pad image to fit to out_size
        int top = -1, down = -1, left = -1, right = -1;
        if(pad){
            top = (static_cast<int>(out_h) - mid_h) / 2;
            down = (static_cast<int>(out_h)- mid_h + 1) / 2;
            left = (static_cast<int>(out_w)- mid_w) / 2;
            right = (static_cast<int>(out_w)- mid_w + 1) / 2;

            cv::copyMakeBorder(dst, dst, top, down, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
        }

        std::vector<float> pad_info{static_cast<float>(left), static_cast<float>(top), scale};
        return pad_info;
    }
    //@}

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

            // print mouth open prob. for now
            float mouthOpenScore = faceFeatures.isMouthOpen();
            float mouthOpenRatio = faceFeatures.getMouthOpenRatio();
            std::cout << "Mouth Open Score = " << mouthOpenScore << std::endl;
            std::cout << "Mouth Open Ratio = " << mouthOpenRatio << std::endl;

            // print smile prob. for now
            float smileScore = faceFeatures.isSmileDetected();
            std::cout << "Smile Score = " << smileScore << std::endl;

            // print eyeglasses prob. for now
            float eyeglassesScore = faceFeatures.isEyeglassesDetected();
            std::cout << "Eyeglasses Score = " << eyeglassesScore << std::endl;
        }
    }
};
#endif // IMAGE_H