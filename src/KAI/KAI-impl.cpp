#include <iostream>
#include <fstream>

// #include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>

#include "KAITaskManager.h"

using json = nlohmann::json;

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "[KAI Task Manager]-- Usage: " << argv[0] << " <image_path> <json_path>" << std::endl;
        return 1;
    }

    // Read image
    std::string imagePath = argv[1];
    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
        std::cerr << "[KAI Task Manager]-- Error: Could not open or find the image!" << std::endl;
        return 1;
    }

    // Read JSON file
    std::string jsonPath = argv[2];
    std::ifstream jsonFile(jsonPath);
    if (!jsonFile.is_open()) {
        std::cerr << "[KAI Task Manager]-- Error: Could not open the MLConfig JSON file!" << std::endl;
        return EXIT_FAILURE;
    }

    // Read output path for image
    std::string outputPath = argv[3];

    // TODO: arg parse option for batch processing
    //       e.g., provide folder and image extensions to process

    // Run KAIWrapper
    KAITaskManager kaiTaskManager_(jsonPath, imagePath);
    kaiTaskManager_.init();

    // diagnostic
    cv::Mat outMat;
    kaiTaskManager_.printOutput(outMat);
    if(!outMat.empty())
        cv::imwrite(outputPath, outMat);

    std::cout << "[KAI Task Manager]-- Process completed successfully!" << std::endl;
    return EXIT_SUCCESS;
}
