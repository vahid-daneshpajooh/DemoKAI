#include "FacePoseEstimator.h"

#include <iostream>

FacePoseEstimator::FacePoseEstimator(const std::string& modelPath,
                const std::string& configPath,
                short backendId, short targetId) {

    // TODO: (thread safety in container envs)
    // cv::dnn::Net is not thread safe, must protect with mutex

    facePoseNet_ = cv::dnn::readNetFromTensorflow(modelPath);
    if (facePoseNet_.empty()){
        // TODO: catch all runtime errors in KAI Task Manager (?)
        throw std::runtime_error("Face Pose Estimation Task -- Error loading the model.");
    }

    /*{ backend Id    | 0 | Choose one of computation backends:
                        "0: automatically (by default), "
                        "1: Halide language (http://halide-lang.org/), "
                        "2: Intel's Deep Learning Inference Engine (https://software.intel.com/openvino-toolkit), "
                        "3: OpenCV implementation }"
      { target  Id    | 0 | Choose one of target computation devices: "
                        "0: CPU target (by default), "
                        "1: OpenCL, "
                        "2: OpenCL fp16 (half-float precision), "
                        "3: VPU }";
    */
   
    facePoseNet_.setPreferableBackend(backendId);
    facePoseNet_.setPreferableTarget(targetId);

    // assign filename to read mean/std values to normalize network inputs
    mMeanStdFilename = configPath;
}

void FacePoseEstimator::init(const std::map<std::string, Type> params)
{
    // TODO: Error handling
    // e.g., 0 < conf < 1 ; width, height > 0 ; 

    // model's input/output layer names
    if (params.find("NNInputName") != params.end()) {
        mInputName = params.at("NNInputName").get<std::string>();
    }
    if (params.find("NNOutputName") != params.end()) {
        mOutputName = params.at("NNOutputName").get<std::string>();
    }

    ///
    // read CSV file containing data normalization values (mean, std.)
    ///
    std::ifstream file (mMeanStdFilename);
    if(!file.is_open()){
        std::string msg = "Face Pose Estimation Task -- "
                          "Error loading the data normalization file: " + mMeanStdFilename;

        // TODO: log it as ERROR

        // TODO: catch all runtime errors in KAI Task Manager (?)
        throw std::runtime_error(msg);
    }

    std::string line;
    int i = 0;
    while (std::getline(file, line))
    {
        std::stringstream row(line);

        // read all elements (separated by comma) in each row
        std::string cell;
        std::vector<float> rowValues;
        while (std::getline(row, cell, ','))
        {
            // convert string to double
            rowValues.push_back(std::stof(cell));
        }

        // copy row values to the vector
        if (i == 0){ // first row: mean
            meanSubtract = rowValues;
        }
        else { // second row: std.
            stdNormalize = rowValues;
        }

        ++i;
    }

    file.close();
}

void FacePoseEstimator::run(Image &img)
{
    // get facial features for all faces detected in Image
    auto vFFeatures = img.getFacialFeatures();
    
    // for each detected face
    for(auto& faceFeature: vFFeatures) {
        // get feature points
        std::vector<cv::Point> featurePoints = faceFeature.getFacialFeatures();
        
        // compute dist vector between all feature pairs
        std::vector<float> distFpairs = computeDistFeaturePairs(featurePoints);

        // Create a cv::Mat object
        // Copy data from the vector to cv::Mat
        cv::Mat matDists(1, nMLFeatures, cv::DataType<float>::type);
        std::memcpy(matDists.data, distFpairs.data(), distFpairs.size() * sizeof(float) );

        std::vector<float> RollYawPitch = {360.0f, 360.0f, 360.0f};
        try {
            facePoseNet_.setInput(matDists, mInputName);
            cv::Mat poseMat = facePoseNet_.forward();

            RollYawPitch[0] = poseMat.at<float>(0,2);
            RollYawPitch[1] = poseMat.at<float>(0,1);
            RollYawPitch[2] = poseMat.at<float>(0,0);
        }
        catch (cv::Exception& e) {
            std::cerr << e.what() << std::endl;
        }   
    }
}

std::vector<float> 
FacePoseEstimator::computeDistFeaturePairs(std::vector<cv::Point> features){
    // NOTES:
    // 1. INITIAL MODEL USED EUCLIDEAN DISTANCE (L-2 norm) BETWEEN EACH PAIRS OF FACIAL FEATURE POINTS
    // (NOT IMPLEMENTED HERE)
    // 2. REVISED MODEL USES DELTA X AND DELTA Y (L-1 norm) BETWEEN EACH PAIR OF FACIAL FEATURE POINTS, (SKIPPING FACE OUTLINE POINTS)
    
    int numFeatues = features.size(); // should be 68 for Dlib

    // compute interocular distance
	int dx = features[43].x - features[37].x;
	int dy = features[43].y - features[37].y;
	float iod = sqrt(dx*dx+dy*dy);
	float iod_scale = 100 / iod;
    
    // compute dist between all possible feature pairs
    // (excluding the face outline; i.e., Dlib landmarks [0, 16])
    std::vector<float> distFPairs;

	int iFeature = 0;
	for  ( int i = 17; i < 68; i++ )	// 17 == index of first feature point following face outline
	{
        float xDist, yDist;
		for ( int j = i + 1; j < 68; j++ )
		{
            // (feature_i, feature_j).x dist
			xDist = static_cast<float>(
                                ( iod_scale * (features[i].x - features[j].x) - meanSubtract[iFeature])
                                / stdNormalize[iFeature]);
			distFPairs.push_back(xDist);
            
            iFeature++;
            // (feature_i, feature_j).y dist
			yDist = static_cast<float>(
                                ( iod_scale * (features[i].y - features[j].y) - meanSubtract[iFeature])
                                / stdNormalize[iFeature]);
			distFPairs.push_back(yDist);
            
            iFeature++;
		}
	}

    return distFPairs;   
}