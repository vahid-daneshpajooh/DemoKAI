#include "KAITaskPipeline.h"
#include <algorithm>

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

    // Execute each task in sequence, passing the bounding boxes
    for (auto& task : taskQueue) {
        task->run(img);
    }
}