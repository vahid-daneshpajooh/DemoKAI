# CMakeLists.txt : CMake project for KAI
# include source and define project specific logic here.
#
cmake_minimum_required (VERSION 3.13)

project(DemoKAI)

# Set C++ standard to 14
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#######################
# 3rd party libraries
#######################

### OpenCV
find_package(OpenCV REQUIRED)


### TensorFlow Lite
set(TFLite_PATH "${CMAKE_CURRENT_SOURCE_DIR}/tensorflow-lite" CACHE PATH
 "Directory that contains the TensorFlow-Lite include and lib" )

set(TFLite_INCLUDE_DIRS "${TFLite_PATH}/include")
set(TFLite_LIBS "${TFLite_PATH}/lib/libtflite.so")


# Add source files
set(SOURCES
	KAI-impl.cpp # main implementation

	KAITaskManager.cpp  # KAI task manager
	KAITaskPipeline.cpp # KAI pipeline

	# KAI tasks
    FaceDetector.cpp
	FacialFeatureDetector.cpp # Dlib model (68 landmarks)
	TFLiteFacialFeatureDetector.cpp # TensorFlow Lite model (468 "Face Mesh" landmarks)

	FacePoseEstimator.cpp
	MouthOpenDetector.cpp
	SmileDetector.cpp
	EyeglassesDetector.cpp

	# Utils
	Logger.cpp
)

# Add header files
set(HEADERS

	KAITaskManager.h   # KAI task manager
	KAITaskPipeline.h  # KAI pipeline
	KAITaskInterface.h # KAI task interface

	# KAI tasks
	FaceDetector.h
	FacialFeatureDetector.h # Dlib model (68 landmarks)
	TFLiteFacialFeatureDetector.h # TensorFlow Lite model (468 "Face Mesh" landmarks)

	FacePoseEstimator.h
	MouthOpenDetector.h
	SmileDetector.h
	EyeglassesDetector.h

	# KAI utils
	MLConfigLoader.h 	# ML Config Loader
	argparser.h 		# input argument parser
	Logger.h
	
	Image.h				# Image class
	FacialFeatures.h	# Facial Features class
	FaceMeshKeypoints.h # map keypoints to facial landmarks
	Types.h				# supported types 

)


# add dlib library as subdirectory
set(DLIB_NO_GUI_SUPPORT ON) # turn off GUI support
set(USE_AVX_INSTRUCTIONS 1) # enable CPU AVX (advanced vector extensions)

add_subdirectory(dlib)

# Add your executable
add_executable(KAI-impl ${SOURCES} ${HEADERS})

# Include OpenCV and TFLite headers
target_include_directories(KAI-impl
	PRIVATE ${OpenCV_INCLUDE_DIRS}
	PRIVATE ${TFLite_INCLUDE_DIRS}
)


# Link OpenCV and TFLite libraries
target_link_libraries(KAI-impl
					PRIVATE ${OpenCV_LIBS}
					PRIVATE dlib::dlib
					PRIVATE ${TFLite_LIBS}
)
