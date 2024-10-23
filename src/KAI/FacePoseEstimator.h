#ifndef FACEPOSEESTIMATOR_H
#define FACEPOSEESTIMATOR_H

#include "KAITaskInterface.h"
#include "FacialFeatures.h"
#include "Types.h"

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

class FacePoseEstimator: public KAITask {
public:
    FacePoseEstimator(const std::string& modelPath, const std::string& configPath,
                short backendId = 0, short targetId = 0);
    
    void init(const std::map<std::string, Type> params);

    void run(Image& img) override;

private:
    cv::dnn::Net facePoseNet_;

    //////////////////
    // network params
    //////////////////
    std::string mInputName = "dense_input";
    std::string mOutputName = "dense_3/MatMul";

    std::string mMeanStdFilename;

    // The number of features excluding the facial outline points.
    // X and Y point distances are 2 different features.
	// 68 == number of facial feature points returned by dlib
	// 17 == number of face outline points that we will not use (indices 0 - 16 inclusive)
	// if we let n = 68 - 17, then the number of feature point pairs is n(n-1)/2
	// however, we now use the x- and y-vectors between each pair of points as ML features, so multiply by 2
	static
    const int nMLFeatures = ((68-17) * (68-17-1) / 2) * 2;	// 2550
    
    std::vector<float> meanSubtract; // [nMLFeatures];
    std::vector<float> stdNormalize; // [nMLFeatures];

    ///
    // helper functions
    ///

    std::vector<float> computeDistFeaturePairs(std::vector<cv::Point> features);
};
#endif // FACEPOSEESTIMATOR_H