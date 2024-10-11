#ifndef KAITASKMANAGER_H
#define KAITASKMANAGER_H

#include <nlohmann/json.hpp>

#include <string>
#include <vector>

// KAI models
#include "FaceDetector.h"

using json = nlohmann::json;

// Define the structure for MLModules
struct MLModule {
    std::string id;
    std::string task;
    int version;
    std::string modelName;
    std::string cfg;
    std::unordered_map<std::string, std::string> vParams;
};

class KAITaskManager {
public:
    KAITaskManager(const std::string configPath, const std::string imagePath);
    
    void init();
    void runFaceDetection();

    void printOutput(cv::Mat& outMat);

private:
    // inputs (image and MLConfig)
    cv::Mat image;
    cv::Mat outImage;
    
    bool outReady = false;

    json MLconfig_;
    std::vector<std::string> vMLConfigIDs;
    std::vector<MLModule> vMLModules;

    void parseMLConfig(json jsonFile);

    FaceDetector* faceDetector;

    // Function to find ML module matching a MLConfig id
    MLModule* findMLModule(std::vector<MLModule>& modules, const std::string& configID);
};

#endif // KAITASKMANAGER_H