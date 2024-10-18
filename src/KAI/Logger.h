#ifndef LOGGER_H
#define LOGGER_H

#include <string>
#include <fstream>
#include <chrono>

enum LogLevel {
    INFO,
    DEBUG,
    ERROR
};

class Logger {
public:
    // Singleton pattern to ensure one logger instance
    static Logger& getInstance();
    
    // Sets the log file output
    void setLogFile(const std::string& fileName);

    // Logs messages at different levels
    void log(LogLevel level, const std::string& message);
    
    // Helper for logging timing information
    void logInferenceTime(const std::string& taskName, 
        const std::chrono::time_point<std::chrono::high_resolution_clock>& startTime);

    // Destructor to close log file
    ~Logger();

private:
    // Private constructor for singleton
    Logger();
    
    std::ofstream logFile;

    // Helper function to get current timestamp
    std::string getCurrentTime() const;

    // Helper function to convert LogLevel to string
    std::string logLevelToString(LogLevel level) const;
};
#endif // LOGGER_H
