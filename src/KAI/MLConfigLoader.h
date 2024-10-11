#ifndef MLCONFIGLOADER_H
#define MLCONFIGLOADER_H

#include <vector>
#include <iostream>
#include <fstream>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

// Define the structure for MLModules
struct MLModule {
    std::string id;
    std::string task;
    int precedence;
    int version;
    std::string modelName;
    std::string cfg;
    std::unordered_map<std::string, std::string> params;
};

class MLConfigLoader {
private:

    std::string jsonFilePath;
    json MLConfig_;

    std::vector<std::string> vMLConfigIDs;
    std::vector<MLModule> vMLModules;

public:
    // Constructor
    MLConfigLoader(const std::string& filePath): jsonFilePath(filePath){
        // load and parse the ML configuration JSON file
        parseMLConfig();
    }

    // Parse MLConfig JSON file
    void parseMLConfig(){

        std::ifstream configFile(jsonFilePath);
        configFile >> MLConfig_;
    
        // reset vectors
        vMLConfigIDs.clear();
        vMLModules.clear();

        vMLConfigIDs = MLConfig_["vMLConfigIDs"];
        
        for (const auto& module : MLConfig_["vMLModules"]) {
            MLModule mlModule;
            mlModule.id = module["id"];
            mlModule.task = module["task"];
            mlModule.precedence = module["precedence"];
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

    // Function to find ML module matching a MLConfig id
    MLModule* findMLModule(std::vector<MLModule>& modules, const std::string& configID) {
        for (auto& module : modules) {
            if (module.id == configID) {
                return &module;
            }
        }
        return nullptr;
    }

    // returns config IDs corresponding to AI tasks
    std::vector<std::string> getMLConfigs(){
        return vMLConfigIDs;
    }

    // returns ML modules corresponding to AI tasks
    std::vector<MLModule> getMLModules(){
        return vMLModules;
    }
};
#endif // MLCONFIGLOADER_H