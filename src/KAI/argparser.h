#include <fstream>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <sys/stat.h>

#include "Logger.h"

using json = nlohmann::json;

std::string outputPath;
std::string jsonPath;
std::string imagePath;

//////////////////////
// heler functions
//////////////////////
bool fileExists(const std::string& path) {
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0);
}

bool isDirectory(const std::string& path) {
    struct stat buffer;
    if (stat(path.c_str(), &buffer) != 0) {
        return false;
    }
    return (buffer.st_mode & S_IFDIR) != 0;
}

int parseArguments(int argc, char** argv){
    
    Logger& logger = Logger::getInstance();

    if (argc < 3) {
        std::string msg = "[KAI Task Manager]-- Usage: " + std::string(argv[0]) + " <image_path> <json_path>";

        // logging
        logger.log(ERROR, msg);

        std::cerr << msg << std::endl;        
        return EXIT_FAILURE;
    }

    // Read image file
    imagePath = argv[1];

    if (!fileExists(imagePath)) {
        std::string msg = "[KAI Task Manager]-- Error: Could not open or find the image!";

        // logging
        logger.log(ERROR, msg);

        std::cerr << msg << std::endl;
        return EXIT_FAILURE;
    }


    // Read JSON file
    jsonPath = argv[2];

    if (!fileExists(jsonPath)) {
        std::string msg = "[KAI Task Manager]-- Error: Could not open the MLConfig JSON file!";

        // logging
        logger.log(ERROR, msg);
        
        std::cerr << msg << std::endl;
        return EXIT_FAILURE;
    }

    // Read output path for image
    outputPath = argv[3];

    // TODO: arg parse option for batch processing
    //       e.g., provide folder and image extensions to process

    return EXIT_SUCCESS;
}

std::string parser_getImagePath(){
    return imagePath;
}

std::string parser_getJSONPath(){
    return jsonPath;
}

std::string parser_getOutputPath(){
    return outputPath;
}