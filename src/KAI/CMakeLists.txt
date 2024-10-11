# CMakeLists.txt : CMake project for KAI
# include source and define project specific logic here.
#
cmake_minimum_required (VERSION 3.13)

project(DemoKAI)

# Set C++ standard to 14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Add source files
set(SOURCES
	KAI-impl.cpp
	KAITaskManager.cpp
    FaceDetector.cpp
)

# Add header files
set(HEADERS
	KAITaskManager.h
	FaceDetector.h
)

# Include OpenCV headers
include_directories(
	${OpenCV_INCLUDE_DIRS}
	)

# Add your executable
add_executable(KAI-impl ${SOURCES} ${HEADERS})

# Link OpenCV libraries
target_link_libraries(KAI-impl ${OpenCV_LIBS})
