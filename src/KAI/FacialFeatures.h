#ifndef FACIALFEATURES_H
#define FACIALFEATURES_H

#include <dlib/image_processing.h>
#include <opencv2/core/types.hpp>

#include <string>
#include <vector>
#include <memory>

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

// Base class for auxiliary data
struct AuxData {

////////////////////////////////
// Auxiliary Data
// 1. Head Pose
// 2. EyesOpen
// 3. Gaze
// 4. MouthOpen
// 5. Smile
// 6. RedEye
///////////////////////////////

    // The main function of FacialFeatures class is to find facial feature points
    // (eyes, nose, mouth, etc.) given a face box. However, there are auxiiary facial data
    // that can be computed and returned at the same time, often using the same data
    // (e.g., input image, facial feature points) already provided by the client.

	enum eAuxDataID		// enum identifying the auxiliary data type
	{
		 eHeadPose	= 0
		,eEyesOpen
		,eGaze
		,eMouthOpen
		,eSmile
		,eRedEye
        ,eEyeglasses
		,eNAuxDataID
	};

    eAuxDataID auxID;
    
    virtual ~AuxData() {}
    AuxData(eAuxDataID id): auxID(id) {}
};

// 1. HeadPose derived from AuxData
struct HeadPose : public AuxData {
    // The roll, pitch, and yaw values are degrees in [-180.0,180)
    // in a right-handed coordinate system with
    // - positive x to the viewer's right, 
    // - positive y down (i.e., the ususal image coordinates),
    // - positive z away from the viewer.
    //   
    // When roll == pitch == yaw, the top of the subject's head is up
	// and the subject is looking out of the image directly at the viewer.
	// 
    // Assuming the world's origin O(x,y,z) is locked to the head of the subject,
    // the Roll, Pitch, and Yaw are defined as follows:
    //
	//	- Roll is rotation around the z axis (the z axis points from the nose to the back of the
	//	  head).  If pitch == yaw == 0, positive roll is clockwise rotation as seen by the viewer.
	//	- Pitch is rotation around the x axis (the x axis points from the subject's right ear to the
	//	  subject's left ear).  If roll == yaw == 0, a small positive pitch causes the face to
	//	  look in the downward direction.
	//	- Yaw is rotation around the y axis (the y axis points from the head to the feet).  If roll ==
	//	  pitch == 0, a small positive yaw causes the face to look to the viewer's left.
    float roll;
    float yaw;
    float pitch;

    HeadPose() : AuxData(eHeadPose), roll(360.0f), yaw(360.0f), pitch(360.0f) {}
};

// 2. EyesOpen derived from AuxData
struct EyesOpen : public AuxData {
    float leftEyeScore;
    float rightEyeScore;

    EyesOpen() : AuxData(eEyesOpen), leftEyeScore(-1.0f), rightEyeScore(-1.0f) {}
};

// 3. Gaze derived from AuxData
struct Gaze : public AuxData {
    float leftEyeYaw;
    float leftEyePitch;
    std::vector<float> leftIrisXYR;

    float rightEyeYaw;
    float rightEyePitch;
    std::vector<float> rightIrisXYR;

    Gaze()
        : AuxData(eGaze), leftEyeYaw(360.0f), leftEyePitch(360.0f), leftIrisXYR(3, 0.0f),
            rightEyeYaw(0.0f), rightEyePitch(0.0f), rightIrisXYR(3, 0.0f) {}
};

// 4. MouthOpen derived from AuxData
struct MouthOpen : public AuxData {
    float openScore;
    float mouthOpenRatio;

    MouthOpen() : AuxData(eMouthOpen), openScore(-1.0f), mouthOpenRatio(-1.0f) {}
};

// 5. Smile derived from AuxData
struct Smile : public AuxData {
    float smileScore;

    Smile() : AuxData(eSmile), smileScore(-1.0f) {}
};

// 6. RedEye derived from AuxData
struct RedEye : public AuxData {
    float leftRedEyeScore;
    float leftEyeFractionRedPixels;

    float rightRedEyeScore;
    float rightEyeFractionRedPixels;

    RedEye() : AuxData(eRedEye), leftRedEyeScore(-1.0f), leftEyeFractionRedPixels(-1.0f),
               rightRedEyeScore(-1.0f), rightEyeFractionRedPixels(-1.0f) {}
};

// 7. Eyeglasses derived from AuxData
struct Eyeglasses : public AuxData {
    float eyeglassesScore;

    Eyeglasses() : AuxData(eEyeglasses), eyeglassesScore(-1.0f) {}
};

class FacialFeatures {
public:
    // Constructor
    FacialFeatures(): smileDetected(false), mouthOpen(false){
        // Initialize all feature locations to default
        for (int i = 0; i < FFeatureLocation::FFNCommonFeatures; ++i) {
            FFlocs.push_back(FFeatureLocation());
        }

        // Initialize all auxiliary data 
        vAuxData.resize(AuxData::eNAuxDataID);
        vAuxData[AuxData::eHeadPose] = std::make_shared<HeadPose>();
        vAuxData[AuxData::eEyesOpen] = std::make_shared<EyesOpen>();
        vAuxData[AuxData::eGaze] = std::make_shared<Gaze>();
        vAuxData[AuxData::eMouthOpen] = std::make_shared<MouthOpen>();
        vAuxData[AuxData::eSmile] = std::make_shared<Smile>();
        vAuxData[AuxData::eRedEye] = std::make_shared<RedEye>();
        vAuxData[AuxData::eEyeglasses] = std::make_shared<Eyeglasses>();
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

    void setFaceBbox(const std::pair<cv::Rect, float>& bbox){
        faceBbox = bbox;
    }
    
    cv::Rect getFaceBbox() const {
        return faceBbox.first;
    }

    /*
    // We set Face Pose as an AuxData now
    void setFacePose(const std::vector<float>& facePose){
        
        RollYawPitch.clear();
        // TODO: assert size is 3 and values in [0, 360)
        RollYawPitch = facePose;
    }
    */

    std::vector<float> getFacePose() const {
        
        // TODO: fix seg fault error
        //       when vAuxData is not initialized with HeadPose
        auto pHeadPose = getAuxData<HeadPose>(AuxData::eHeadPose);
        
        std::vector<float> RollYawPitch = {360.0f, 360.0f, 360.0f};
        RollYawPitch[0] = pHeadPose->roll;
        RollYawPitch[1] = pHeadPose->yaw;
        RollYawPitch[2] = pHeadPose->pitch;

        return RollYawPitch;
    }

    float isMouthOpen() const {
        auto pMouthOpen = getAuxData<MouthOpen>(AuxData::eMouthOpen);

        return pMouthOpen->openScore;
    }
    float getMouthOpenRatio() const {
        auto pMouthOpen = getAuxData<MouthOpen>(AuxData::eMouthOpen);

        // compute MouthOpen Ratio (if not available) before returning aux data
        if(pMouthOpen->mouthOpenRatio == -1){
            const float ratio = computeMouthOpenRatioDlib();
            pMouthOpen->mouthOpenRatio = ratio;
        }

        return pMouthOpen->mouthOpenRatio;
    }

    float isSmileDetected() const {
        auto pSmile = getAuxData<Smile>(AuxData::eSmile);

        return pSmile->smileScore;
    }

    float isEyeglassesDetected() const {
        auto pEyeglasses = getAuxData<Eyeglasses>(AuxData::eEyeglasses);

        return pEyeglasses->eyeglassesScore;
    }

    // Methods to add or retrieve auxiliary data
    template<typename T>
    void setAuxData(std::shared_ptr<T> auxData) {
        vAuxData[auxData->auxID] = auxData;
    }

    template<typename T>
    std::shared_ptr<T> getAuxData(AuxData::eAuxDataID auxID) const {
        return std::dynamic_pointer_cast<T>(vAuxData[auxID]);
    }

    // Methods to handle additional features (e.g., smile detection, eye state)

    float getIOD() {
        return computeIOD();
    }

    // Method to clear all stored features (for reprocessing)
    void clearFeatures();

    // Utility to get a string representation of all features (for logging)
    std::string getFeatureSummary() const;

private:
    // face bounding box and confidence score
    std::pair<cv::Rect, float> faceBbox;

    // facial feature points
    // e.g., 68 for Dlib model
    std::vector<cv::Point> vFFpoints;

    // Facial landmarks (eyes, nose, mouth)
    std::vector<FFeatureLocation> FFlocs;

    // Auxiliary facial features
    
    // vector to store all auxiliary data
    std::vector<std::shared_ptr<AuxData>> vAuxData;

    // 1. Head Pose (Roll, Yaw, and Pitch)
    // param range: [0, 360) degrees
    // (360 is invalid value)

    // 4. Mouth Open
    bool mouthOpen;

    // 5. Smile Detected
    bool smileDetected;

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
        FFlocs[FFeatureLocation::FFMouthTop] = FFeatureLocation(landmarks.part(51).x(), landmarks.part(51).y());
        FFlocs[FFeatureLocation::FFNoseLeftSide] = FFeatureLocation(landmarks.part(31).x(), landmarks.part(31).y());
        FFlocs[FFeatureLocation::FFNoseRightSide] = FFeatureLocation(landmarks.part(35).x(), landmarks.part(35).y());

        // TODO: fill eyes and mouth centers;
        // e.g., LeftEyeCenter  = [avg(x), avg(y)]_(36, 37, 38, 39, 40, 41) (Centroid ~ Mean Center)
        //       RightEyeCenter = [avg(x), avg(y)]_(42, 43, 44, 45, 46, 47) (Centroid ~ Mean Center)
        //       MouthCenter    = average(62, 66)    (?)

        // left eye
        float centroid_x, centroid_y;
        for (int i = 36; i < 42; ++i){
            centroid_x += landmarks.part(i).x();
            centroid_y += landmarks.part(i).y();
        }
        FFlocs[FFeatureLocation::FFLeftEyeCenter] = FFeatureLocation(centroid_x / (42-36), centroid_y / (42-36));

        // right eye
        centroid_x = 0;
        centroid_y = 0;
        for (int i = 42; i < 48; ++i){
                centroid_x += landmarks.part(i).x();
                centroid_y += landmarks.part(i).y();
        }
       FFlocs[FFeatureLocation::FFRightEyeCenter] = FFeatureLocation(centroid_x / (48-42), centroid_y / (48-42));

        // mouth
        centroid_x = (landmarks.part(62).x() + landmarks.part(66).x()) / 2;
        centroid_y = (landmarks.part(62).y() + landmarks.part(66).y()) / 2;
        FFlocs[FFeatureLocation::FFMouthCenter] = FFeatureLocation(centroid_x, centroid_y);
    }

    ////////////////////////
    // helper methods
    ///////////////////////
    
    float computeIOD(){
        // left eye center
        float Cx_leftEye = FFlocs[FFeatureLocation::FFLeftEyeCenter].mX;
        float Cy_leftEye = FFlocs[FFeatureLocation::FFLeftEyeCenter].mY;
        cv::Point2f C_leftEye(Cx_leftEye, Cy_leftEye);

        // right eye center
        float Cx_rightEye = FFlocs[FFeatureLocation::FFRightEyeCenter].mX;
        float Cy_rightEye = FFlocs[FFeatureLocation::FFRightEyeCenter].mY;
        cv::Point2f C_rightEye(Cx_rightEye, Cy_rightEye);

        return cv::norm(C_rightEye - C_leftEye); // euclidean dist between left and right eyes
    }

    float computeMouthOpenRatioDlib() const {
    
        float mouthOpenRatio = -1.0f;
        
        // vector from left to right mouth corner
        float LipLineDx, LipLineDy;
        // mouth corner to corner dist
        float corner2corner;
        // average of upper to lower lip distances
        float MouthLipDist = 0.0;

        LipLineDx = FFlocs[FFeatureLocation::FFMouthRightCorner].mX
                    - FFlocs[FFeatureLocation::FFMouthLeftCorner].mX;
        LipLineDy = FFlocs[FFeatureLocation::FFMouthRightCorner].mY
                    - FFlocs[FFeatureLocation::FFMouthLeftCorner].mY;
        
        cv::Point lipLineVec(LipLineDx, LipLineDy);
        corner2corner = cv::norm(lipLineVec);

        int iUpLip,iLoLip,iUpLipLast;
        // for Dlib facial feature points
        iUpLip = 61, iLoLip = 67, iUpLipLast = 63;
        for  ( ; iUpLip <= iUpLipLast; iUpLip++, iLoLip-- )
        {
            // vector from upper lip to lower lip
            cv::Point mouthVec;
            mouthVec = vFFpoints[iLoLip] - vFFpoints[iUpLip];

            // to be valid, mouth vec CROSS lip vec should be positive, otherwise it counts as zero distance
            if  ( lipLineVec.cross(mouthVec) > 0.0 )
                MouthLipDist += cv::norm(mouthVec);
        }
        
        // TODO: 3 is for averaging over three top-bottom Fpoints dists? what if one is invalid? why divide by 3?
        // avoid divide by zero case (FFeatureLocation = 0.0)
        if(corner2corner != 0 ){
            mouthOpenRatio = MouthLipDist / (3.0 * corner2corner);
        }

        return mouthOpenRatio;
    }


};
#endif // FACIALFEATURES_H