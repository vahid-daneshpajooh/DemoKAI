#ifndef IMAGE_H
#define IMAGE_H

#include <vector>
#include <opencv2/opencv.hpp>

class Image {
public:
    Image(const std::string& img_path){
        imgMat = cv::imread(img_path);
    }

    void getImage_Mat(cv::Mat& outMat){
        imgMat.copyTo(outMat);
    }

    std::vector<cv::Rect> getImage_faceBboxes(){
        return faceBboxes;
    }

    void setImage_faceBboxes(const std::vector<cv::Rect> bboxes){
        // clear current vector elements
        faceBboxes.clear();

        faceBboxes = bboxes;
    }
    
    void getImage_faceOn(cv::Mat& outMat){
        if (imgMat.empty()){
            return;
        }

        outMat = imgMat.clone();
        overlayFaceOnImage(outMat);
    }

private:
    cv::Mat imgMat;

    // Store face bounding boxes (x, y, width, height)
    std::vector<cv::Rect> faceBboxes;


    // util functions
    void overlayFaceOnImage(cv::Mat& overlayImg){
        for(const auto& faceBox: faceBboxes){
            cv::rectangle(overlayImg, faceBox, cv::Scalar(45, 255, 45), 2, 4);
            // TODO: requires saving confidence score in Image when running AI Task
            // cv::putText(imgMat, std::to_string(static_cast<int>(confidence*100)),
            //cv::Point(x1+2, y1-5), cv::FONT_HERSHEY_SIMPLEX, 0.5, (45, 45, 255), 2);
        }
    }
};
#endif // IMAGE_H
