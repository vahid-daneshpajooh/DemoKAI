#ifndef TYPES_H
#define TYPES_H

#include <string>
#include <vector>
#include <variant> // requires C++ 17

// Define a variant type for supported parameter types
using supportTypes = std::variant<int, float, double, std::string,
                                std::vector<float>, std::vector<double>
                                >;


// Structure to hold a supported parameter's value and type
struct Type
{
    supportTypes value; // parameter value
    std::string type; // parameter type

    ///
    // Helper functions
    ///

    // helper to check if value is of a certain type
    template<typename T>
    bool isSupportType() const {
        return std::holds_alternative<T>(value);
    }

    // helper to get value safely as a certain type
    template<typename T>
    T get() const {
        return std::get<T>(value);
    }
};
#endif // TYPES_H