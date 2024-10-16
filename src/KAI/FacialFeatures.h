#ifndef FACIALFEATURES_H
#define FACIALFEATURES_H

#include <dlib/image_processing.h>
#include <opencv2/core/types.hpp>

#include <string>

struct FFeatureLocation
{
/*
=============================
Dlib facial landmarks
=============================
0-based facial feature points:

         18 19 20              23 24 25
     17            21      22            26

           37 38                 43 44
       36         39   27    42         45
00         41 40                 47 46         16
                       28

01                     29                      15

                       30
 02              31          35               14
                    32 33 34

  03                                         13

                    50 51 52
    04          49  61 62 63  53           12
            48  60            64  54
                59  67 66 65  55
       05           58 57 56            11


          06                         10

              07                09
                       08
*/
// enum "eFFIndex": index to an array element containing the index of the corresponding facial feature point.
// Note: -1 indicates the facial feature finder does not provide the corresponding point.
//
// Example:
//
	enum eFFIndex
	{
		 FFNotAvailable = -1,
         FFLeftEyeCenter,
         FFLeftEyeLeftCorner,
         FFLeftEyeRightCorner,
         FFRightEyeCenter,
         FFRightEyeLeftCorner,
         FFRightEyeRightCorner,
         FFNoseLeftSide,
         FFNoseRightSide,
         FFMouthCenter,
         FFMouthLeftCorner,
         FFMouthRightCorner,
         FFMouthTop,
         FFNCommonFeatures
	};

	// facial feature coordinates and (optional) confidence level
	float	mX;
	float	mY;
	
    // Confidence may be [0,1],
    // but also may be an arbitrary number (e.g., Omron);
    // -1 for unknown
	float	mConfidence;

	/*<constructor>*/	FFeatureLocation  (
		) :
		mX(0.0),
		mY(0.0),
		mConfidence(-1.0)
	{
	}

	/*<constructor>*/	FFeatureLocation  (
		float	x,
		float	y
		) :
		mX(x),
		mY(y),
		mConfidence(-1.0)
	{
	}

	/*<constructor>*/	FFeatureLocation  (
		float	x,
		float	y,
        float conf
		) :
		mX(x),
		mY(y),
		mConfidence(conf)
	{
	}
};

class FacialFeatures {
public:
    // Constructor
    FacialFeatures() : smileDetected(false) {
        // Initialize all feature locations to default
        for (int i = 0; i < FFeatureLocation::FFNCommonFeatures; ++i) {
            FFlocs.push_back(FFeatureLocation());
        }
    }

    // landmarks, including (eye, mouth, nose) left/right corners and center
    std::vector<FFeatureLocation>  getFacialLandmarks() const {
        return FFlocs;
    };

    // get all available facial feature points
    std::vector<cv::Point> getFacialFeatures() const {
        return vFFpoints;
    }

    void setFFeaturesFromDlib(const dlib::full_object_detection& landmarks){
        
        vFFpoints.clear();
        for (int i = 0; i < landmarks.num_parts(); ++i) {
            vFFpoints.push_back(cv::Point(landmarks.part(i).x(), landmarks.part(i).y()));
        }

        // save specific FFpoints as facial landmarks
        setFacialLandmarksFromDlib(landmarks);
    }

    // Methods to handle additional features (e.g., smile detection, eye state)
    void setSmileDetected(bool smile);
    bool isSmileDetected() const;

    // Method to clear all stored features (for reprocessing)
    void clearFeatures();

    // Utility to get a string representation of all features (for logging)
    std::string getFeatureSummary() const;

private:
    // Other possible facial features
    bool smileDetected;

    // facial feature points
    // e.g., 68 for Dlib model
    std::vector<cv::Point> vFFpoints;

    // Facial landmarks (eyes, nose, mouth)
    std::vector<FFeatureLocation> FFlocs;

    ///
    // helper functions
    ///

    // Map the specific landmark points to facial feature locations
    void setFacialLandmarksFromDlib(const dlib::full_object_detection& landmarks) {
        FFlocs[FFeatureLocation::FFLeftEyeLeftCorner] = FFeatureLocation(landmarks.part(36).x(), landmarks.part(36).y());
        FFlocs[FFeatureLocation::FFLeftEyeRightCorner] = FFeatureLocation(landmarks.part(39).x(), landmarks.part(39).y());
        FFlocs[FFeatureLocation::FFRightEyeLeftCorner] = FFeatureLocation(landmarks.part(42).x(), landmarks.part(42).y());
        FFlocs[FFeatureLocation::FFRightEyeRightCorner] = FFeatureLocation(landmarks.part(45).x(), landmarks.part(45).y());
        FFlocs[FFeatureLocation::FFMouthLeftCorner] = FFeatureLocation(landmarks.part(48).x(), landmarks.part(48).y());
        FFlocs[FFeatureLocation::FFMouthRightCorner] = FFeatureLocation(landmarks.part(54).x(), landmarks.part(54).y());
        FFlocs[FFeatureLocation::FFNoseLeftSide] = FFeatureLocation(landmarks.part(31).x(), landmarks.part(31).y());
        FFlocs[FFeatureLocation::FFNoseRightSide] = FFeatureLocation(landmarks.part(35).x(), landmarks.part(35).y());
    }
};

#endif // FACIALFEATURES_H
