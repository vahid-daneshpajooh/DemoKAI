#include <iostream>
#include <fstream>

#include "argparser.h"
#include "Logger.h"

#include "Image.h"

#include "KAITaskManager.h"

// using json = nlohmann::json;

int main(int argc, char** argv) {
    
    // TODO: add logger as input options --log
    // Initialize logger
    // (can set to a file or keep it as console output)
    Logger& logger = Logger::getInstance();
    // Optional: log to a file
    logger.setLogFile("pipeline_log.txt");

    int Failed = parseArguments(argc, argv);
    if (Failed) {
        return EXIT_FAILURE;
    }

    std::string img_path = parser_getImagePath();
    std::string json_path = parser_getJSONPath();

    // TODO assert that it was provided
    std::string output_path = parser_getOutputPath();

    // Run KAI Task Manager
    KAITaskManager kaiTaskManager;
    kaiTaskManager.loadMLConfigs(json_path);

    Image img(img_path);
    kaiTaskManager.runTasks(img);
    
    // print results on image
    cv::Mat outMat;
    
    // draw faces
    img.getImage_faceOn(outMat);
    // draw facial landmarks
    img.getImage_faceFeaturesOn(outMat);

    if(!outMat.empty() && !output_path.empty())
        cv::imwrite(output_path, outMat);

    std::string msg = "[KAI Task Manager]-- Process completed successfully!"
                      "\n===================================================";

    logger.log(INFO, msg);

    std::cout << msg << std::endl;
    return EXIT_SUCCESS;
}
