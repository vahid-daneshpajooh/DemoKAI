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
	FacialFeatureDetector.cpp
)

# Add header files
set(HEADERS

	KAITaskManager.h   # KAI task manager
	KAITaskPipeline.h  # KAI pipeline
	KAITaskInterface.h # KAI task interface

	# KAI tasks
	FaceDetector.h
	FacialFeatureDetector.h

	# KAI utils
	MLConfigLoader.h 	# ML Config Loader
	argparser.h 		# input argument parser
	
	Image.h				# Image class
	FacialFeatures.h
	Types.h				# supported types 

)


# add dlib library as subdirectory
set(DLIB_NO_GUI_SUPPORT ON) # turn off GUI support
set(USE_AVX_INSTRUCTIONS 1) # enable CPU AVX (advanced vector extensions)

add_subdirectory(dlib)


# Include OpenCV headers
include_directories(
	${OpenCV_INCLUDE_DIRS}
	)

# Add your executable
add_executable(KAI-impl ${SOURCES} ${HEADERS})

# Link OpenCV libraries
target_link_libraries(KAI-impl ${OpenCV_LIBS} dlib::dlib)
