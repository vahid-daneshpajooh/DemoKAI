#include "KAITaskManager.h"
#include "FaceDetector.h"

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
        if(str_task == "FD"){
            // read model and cfg files (*.caffemodel and *.prototxt)
            std::string modelPath = module.modelName;
            std::string cfgPath = module.cfg;

            // TODO: read other task specific params;
            // 1) confidence level1 and level2,
            // 2) input image h and w,
            // 3) input and output layer names,
            // 4) image mean subtraction array,
            // ...

            task = std::make_unique<FaceDetector>(modelPath, cfgPath);
        }
        else if(str_task == "FF"){
            // TODO: add facial feature point extraction class
        }

        if(task){
            task->setPrecedence(module.precedence);
            kai_pipeline.addTask(std::move(task));
        }
    }
}

void KAITaskManager::runTasks(Image& img){
    kai_pipeline.runPipeline(img);
}