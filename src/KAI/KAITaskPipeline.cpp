#include "KAITaskPipeline.h"
#include "Logger.h"

#include <algorithm>
#include <chrono>

// Add task to the pipeline
void KAITaskPipeline::addTask(std::unique_ptr<KAITask> task) {
    taskQueue.push_back(std::move(task));
}

// Sort the tasks based on their priority
void KAITaskPipeline::sortTasksByPriority() {
    std::sort(taskQueue.begin(), taskQueue.end(),
            [](const std::unique_ptr<KAITask>& a, const std::unique_ptr<KAITask>& b)
            {
                return a->getPrecedence() < b->getPrecedence();
            });
}

// Execute all tasks in the sorted order
void KAITaskPipeline::runPipeline(Image& img) {

    // Sort the tasks before executing
    sortTasksByPriority();

    // logging 
    Logger& logger = Logger::getInstance();
    logger.log(INFO, "Processing image: " + img.getName());

    // Execute each task in sequence, passing the bounding boxes
    for (auto& task : taskQueue) {
        // logging - [task name]
        logger.log(INFO, "Starting task: " + task->getName());
        
        auto startTime = std::chrono::high_resolution_clock::now();
        
        // Run task
        // TODO: handle errors at top-level (?)
        task->run(img);       
        
        // logging - [task inference time]
        logger.logInferenceTime(task->getName(), startTime);
    }
}