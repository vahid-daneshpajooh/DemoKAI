# CMakeLists.txt : CMake project for KAI
# include source and define project specific logic here.
#
cmake_minimum_required (VERSION 3.13)

project(DemoKAI)

# Set C++ standard to 14
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Add source files
set(SOURCES
	KAI-impl.cpp # main implementation

	KAITaskManager.cpp  # KAI task manager
	KAITaskPipeline.cpp # KAI pipeline

	# KAI tasks
    FaceDetector.cpp
)

# Add header files
set(HEADERS

	KAITaskManager.h   # KAI task manager
	KAITaskPipeline.h  # KAI pipeline
	KAITaskInterface.h # KAI task interface

	# KAI tasks
	FaceDetector.h

	# KAI utils
	MLConfigLoader.h 	# ML Config Loader
	argparser.h 		# input argument parser
	Image.h				# Image class
	Types.h				# supported types 

)

# Include OpenCV headers
include_directories(
	${OpenCV_INCLUDE_DIRS}
	)

# Add your executable
add_executable(KAI-impl ${SOURCES} ${HEADERS})

# Link OpenCV libraries
target_link_libraries(KAI-impl ${OpenCV_LIBS})
