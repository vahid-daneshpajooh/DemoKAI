#include "KAITaskManager.h"

#include <iostream>
#include <fstream>
#include <algorithm>

KAITaskManager::KAITaskManager(const std::string configPath,
                        const std::string imagePath) {

    std::ifstream configFile(configPath);
    if (!configFile.is_open()) {
        throw std::runtime_error("Could not open config file");
    }

    configFile >> MLconfig_;
    parseMLConfig(MLconfig_);

    image = cv::imread(imagePath);

    faceDetector = nullptr;
}

void KAITaskManager::init() {
    
    if(vMLConfigIDs.empty()){
        return;
    }
    
    auto iter_FD = std::find_if(vMLModules.begin(), vMLModules.end(), [](const auto& module){
                                    return module.task == "FD";
                                });
    
    if(iter_FD != vMLModules.end()){

        std::string modelFile = iter_FD->modelName;
        std::string modelCfg = iter_FD->cfg;

        faceDetector = new FaceDetector(modelFile, modelCfg);
        
        runFaceDetection();
    }
    else {
        std::cerr << "[KAI Task Manager]-- Unsupported task: " << iter_FD->task << std::endl;
    }
}

void KAITaskManager::runFaceDetection() {

    // TODO: read params from config file and set in faceDetector

    faceDetector->detectFaces(image);

    outReady = true;
}

void KAITaskManager::parseMLConfig(json jsonFile)
{
    // Parse MLConfig JSON file
    vMLConfigIDs = jsonFile["vMLConfigIDs"];
    
    for (const auto& module : jsonFile["vMLModules"]) {
        MLModule mlModule;
        mlModule.id = module["id"];
        mlModule.task = module["task"];
        mlModule.version = module["version"];
        mlModule.modelName = module["modelName"];
        mlModule.cfg = module["cfg"];

        // TODO 1: some keys are unique to a module, what to do?
        // e.g., cfg file (.prototxt) for face detection Caffe model

        // TODO 2: parameter types vary case-by-case
        /*
        for (const auto& param : module["vParams"].items()) {
            mlModule.vParams[param.key()] = param.value();
        }
        */
        vMLModules.push_back(mlModule);
    }

    // Find and print the MLModule corresponding to each vMLConfigID
    for (const auto& configID : vMLConfigIDs) {
        MLModule* module = findMLModule(vMLModules, configID);
        if (module) {
            std::cout << "Found MLModule for configID " << configID << ":\n";
            std::cout << "--id: " << module->id << "\n";
            std::cout << "--task: " << module->task << "\n";
            std::cout << "--version: " << module->version << "\n";
            /*
            std::cout << "\nParameters: ";
            for (const auto& param : module->vParams) {
                std::cout << param.first << ": " << param.second << " ";
            }
            */

            std::cout << "\n\n";
        } 
        else {
            std::cout << "No MLModule found for configID " << configID << "\n";
        }
    }
}

MLModule* KAITaskManager::findMLModule(std::vector<MLModule>& modules, const std::string& configID) {
    for (auto& module : modules) {
        if (module.id == configID) {
            return &module;
        }
    }
    return nullptr;
}

void KAITaskManager::printOutput(cv::Mat &outMat)
{
    if (outReady)
        image.copyTo(outMat);

    return;

}