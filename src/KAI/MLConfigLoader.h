#ifndef MLCONFIGLOADER_H
#define MLCONFIGLOADER_H

#include <vector>
#include <iostream>
#include <fstream>

#include <nlohmann/json.hpp>

#include "Types.h"

using json = nlohmann::json;

// Define the structure for MLModules
struct MLModule {
    std::string id;
    std::string task;
    int precedence;
    int version;
    std::string modelName;  // model path
    std::string cfg;        // model config file (if any)
    std::map<std::string, Type> params; // task-specific params
};

class MLConfigLoader {
private:

    std::string jsonFilePath;
    json MLConfig_;

    std::vector<std::string> vMLConfigIDs;
    std::vector<MLModule> vMLModules;

    // parse vector types
    std::vector<float> parseVector(const std::string& vecStr) {
        
        // supports values of type double
        // (and implicit cast from int, float, etc.)
        std::vector<float> vec;
        
        std::stringstream ss(vecStr);
        std::string temp;
        
        // vecStr template: "[ 104.0, 177.0, 123.0 ]"
        // Remove the brackets and commas
        ss.ignore(1, '[');
        while (std::getline(ss, temp, ',')) {
            vec.push_back(std::stof(temp));
        }

        return vec;
    }


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
            /* loading params keys:values
            for(auto& item: module["vParams"].items()){
                mlModule.params[item.key()] = item.value().dump();
            }
            */

            for (auto& item: module["vParams"].items()) {
                
                // read param (value, type)
                
                // template: "paramName: ["value", "type"]"
                const std::string& paramName = item.key();
                const auto& value_and_type = item.value(); // value and type
                
                // read type as string
                std::string type = value_and_type[1];

                // Set the type and value based on the "type" string
                Type paramValue;
                if (type == "int") {
                    paramValue.value = value_and_type[0].get<int>();
                }
                else if (type == "float") {
                    paramValue.value = value_and_type[0].get<float>();
                }
                else if (type == "double") {
                    paramValue.value = value_and_type[0].get<double>();
                }
                else if (type == "string") {
                    type.insert(0, "std::");
                    paramValue.value = value_and_type[0].get<std::string>();
                }
                else if (type == "vector<float>" || type == "vector<double>") {
                    type.insert(0, "std::");
                    // Parse vector from a string representation to vector<double>
                    paramValue.value = parseVector(value_and_type[0].get<std::string>());
                }

                paramValue.type = type;
                mlModule.params[paramName] = paramValue;
            }
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