#include "KAITaskManager.h"
#include "FaceDetector.h"
#include "FacialFeatureDetector.h"

#include <iostream>
#include <fstream>
#include <algorithm>

void KAITaskManager::loadMLConfigs(const std::string config_path)
{
    MLConfigLoader configLoader(config_path);
    
    // do nothing if no task is defined
    if(configLoader.getMLConfigs().empty()){
        return;
    }

    auto vMLModules = configLoader.getMLModules();

    // TODO: add tasks to the pipeline here
    for(const auto& module: vMLModules){
        std::unique_ptr<KAITask> task;

        std::string str_task = module.task;
        if(str_task == "FaceDetection"){
            // read model and cfg files (*.caffemodel and *.prototxt)
            std::string modelPath = module.modelName;
            std::string cfgPath = module.cfg;
            
            // pass model to constructor
            FaceDetector* pFaceDetector = new FaceDetector(modelPath, cfgPath);

            // read other task specific params and initialize model
            auto params = module.params;
            pFaceDetector->init(params);

            task = std::unique_ptr<KAITask>(pFaceDetector);
        }
        else if(str_task == "FacialFeatures"){
            // read model Dlib file (*.dat)
            std::string modelPath = module.modelName;
            // pass model to constructor
            task = std::make_unique<FacialFeatureDetector>(modelPath);
        }

        if(task){
            task->setName(module.task);
            task->setPrecedence(module.precedence);
            kai_pipeline.addTask(std::move(task));
        }
    }
}

void KAITaskManager::runTasks(Image& img){
    kai_pipeline.runPipeline(img);
}