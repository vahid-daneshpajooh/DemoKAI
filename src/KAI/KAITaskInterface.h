#ifndef KAITASKINTERFACE_H
#define KAITASKINTERFACE_H

#include "Image.h"

class KAITask{
public:

    virtual ~KAITask() = default;

    virtual void run(Image& img) = 0;
    
    // Get task name
    virtual std::string getName() const {return taskName;}
    
    // Set task name
    virtual void setName(const std::string name) {taskName = name;}
    
    // Get task priority
    virtual int getPrecedence() const {return precedence;}

    // Set task's priority
    virtual void setPrecedence(int num) {precedence = num;}

private:

    int precedence; // task precedence (lower value = higher priority)
    
    std::string taskName; // task name
};

#endif // KAITASKINTERFACE_H