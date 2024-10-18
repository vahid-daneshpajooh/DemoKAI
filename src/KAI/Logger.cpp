#include "Logger.h"
#include <iostream>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <chrono>

Logger& Logger::getInstance() {
    static Logger instance;  // Guaranteed to be destroyed and instantiated on first use
    return instance;
}

Logger::Logger() {}

void Logger::setLogFile(const std::string& fileName) {
    if (logFile.is_open()) {
        logFile.close();
    }
    logFile.open(fileName, std::ios_base::app);
}

Logger::~Logger() {
    if (logFile.is_open()) {
        logFile.close();
    }
}

void Logger::log(LogLevel level, const std::string& message) {
    std::string logMessage = "[" + getCurrentTime() + "] " + logLevelToString(level) + ": " + message;
    
    if (logFile.is_open()) {
        logFile << logMessage << std::endl;
    }
    else {
        std::cout << logMessage << std::endl;
    }
}

void Logger::logInferenceTime(const std::string& taskName, 
        const std::chrono::time_point<std::chrono::high_resolution_clock>& startTime) {
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    std::string message = "Task: " + taskName + " completed in " + std::to_string(duration) + " ms";
    log(INFO, message);
}

std::string Logger::getCurrentTime() const {
    auto now = std::chrono::system_clock::now();
    std::time_t nowTime = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&nowTime), "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

std::string Logger::logLevelToString(LogLevel level) const {
    switch (level) {
        case INFO: return "INFO";
        case DEBUG: return "DEBUG";
        case ERROR: return "ERROR";
        default: return "UNKNOWN";
    }
}
