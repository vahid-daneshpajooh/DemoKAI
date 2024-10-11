#ifndef KAITASKPIPELINE_H
#define KAITASKPIPELINE_H

#include <vector>
#include <memory>

#include "KAITaskInterface.h"

class KAITaskPipeline {
private:
    std::vector<std::unique_ptr<KAITask>> taskQueue;

public:
    void addTask(std::unique_ptr<KAITask> task);
    void runPipeline(Image& img);
    void sortTasksByPriority();
};
#endif // KAITASKPIPELINE_H