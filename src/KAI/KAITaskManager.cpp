#include "KAITaskManager.h"

#include "FaceDetector.h"
#include "FacialFeatureDetector.h"
#include "TFLiteFacialFeatureDetector.h"

#include "FacePoseEstimator.h"
#include "MouthOpenDetector.h"
#include "SmileDetector.h"
#include "EyeglassesDetector.h"

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
        else if(str_task == "FacialFeatures" && module.id == "FFDefault"){
            // read Dlib model file (*.dat)
            std::string modelPath = module.modelName;
            
            // pass model to constructor
            task = std::make_unique<FacialFeatureDetector>(modelPath);
        }
        else if(str_task == "FacialFeatures" && module.id == "FFTFlowLite"){
            // read tensorflow lite model file (*.tflite)
            std::string modelPath = module.modelName;
            
            // pass model to constructor
            TFLiteFacialFeatureDetector* pFacialFeatureDetector = 
                                        new TFLiteFacialFeatureDetector(modelPath);

            // read other task specific params and initialize model
            auto params = module.params;
            pFacialFeatureDetector->init(params);

            // pass model to constructor
            task = std::unique_ptr<KAITask>(pFacialFeatureDetector);
        }
        else if(str_task == "FacePose"){
            // read model and cfg files (*.pb and *.csv)
            std::string modelPath = module.modelName;
            std::string cfgPath = module.cfg;
            
            // pass model to constructor
            FacePoseEstimator* pFacePoseEstimator = new FacePoseEstimator(modelPath, cfgPath);

            // read other task specific params and initialize model
            auto params = module.params;
            pFacePoseEstimator->init(params);

            task = std::unique_ptr<KAITask>(pFacePoseEstimator);
        }
        else if (str_task == "MouthOpen"){
            // read model filename (*.pb)
            std::string modelPath = module.modelName;

            // pass model to constructor
            MouthOpenDetector* pMouthOpenDetector = new MouthOpenDetector(modelPath);

            // read other task specific params and initialize model
            auto params = module.params;
            pMouthOpenDetector->init(params);

            task = std::unique_ptr<KAITask>(pMouthOpenDetector);
        }
        else if (str_task == "Smile"){
            // read model filename (*.pb)
            std::string modelPath = module.modelName;

            // pass model to constructor
            SmileDetector* pSmileDetector = new SmileDetector(modelPath);

            // read other task specific params and initialize model
            auto params = module.params;
            pSmileDetector->init(params);

            task = std::unique_ptr<KAITask>(pSmileDetector);
        }
        else if (str_task == "Eyeglasses"){
            // read model filename (*.pb)
            std::string modelPath = module.modelName;

            // pass model to constructor
            EyeglassesDetector* pEyeglassesDetector = new EyeglassesDetector(modelPath);

            // read other task specific params and initialize model
            auto params = module.params;
            pEyeglassesDetector->init(params);

            task = std::unique_ptr<KAITask>(pEyeglassesDetector);
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