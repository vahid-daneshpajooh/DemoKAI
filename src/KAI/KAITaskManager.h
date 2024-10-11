#ifndef KAITASKMANAGER_H
#define KAITASKMANAGER_H

// #include <nlohmann/json.hpp>

#include <string>
#include <vector>

#include "MLConfigLoader.h"
#include "KAITaskPipeline.h"


// using json = nlohmann::json;

class KAITaskManager {
public:
    //KAITaskManager();
    
    void loadMLConfigs(const std::string config_path);

    void runTasks(Image& image);

private:
    
    KAITaskPipeline kai_pipeline;
};

#endif // KAITASKMANAGER_H