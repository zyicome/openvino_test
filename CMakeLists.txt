# cmake needs this line
cmake_minimum_required(VERSION 3.10)
# Define project name
project(openvino_test_two)
# Find OpenCV, you may need to set OpenCV_DIR variable
# to the absolute path to the directory containing OpenCVConfig.cmake file
# via the command line or GUI
set(OpenVINO_DIR "/opt/openvino/runtime/cmake")
find_package(OpenVINO REQUIRED)
find_package(OpenCV REQUIRED)
# If the package has been found, several variables will
# be set, you can find the full list with descriptions
# in the OpenCVConfig.cmake file.
# Print some message showing some of them
# set "-std=c++11"
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

set(OpenVINO_LIBRARIES "/opt/openvino/runtime/lib/intel64/libopenvino.so")
set(OpenVINO_INCLUDE_DIRS "/opt/openvino/runtime/include")

# Add OpenCV headers location to your include paths
# include_directories(${OpenCV_INCLUDE_DIRS})

# Declare the executable target built from your sources
add_executable(openvino_test_two openvino-infer.cpp)
# Link your application with OpenCV libraries
target_link_libraries(openvino_test_two PRIVATE ${OpenCV_LIBS} ${OpenVINO_LIBRARIES})
target_include_directories(openvino_test_two PRIVATE ${OpenCV_INCLUDE_DIRS} ${OpenVINO_INCLUDE_DIRS})
