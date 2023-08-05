cmake_minimum_required(VERSION 3.25)
project(AthleteDetectionProject)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV 4.2 REQUIRED)

add_executable( AthleteDetectionProject main.cpp
                processingdata.h    processingdata.cpp
                commons.h)

target_link_libraries(AthleteDetectionProject ${OpenCV_LIBS})
target_include_directories(AthleteDetectionProject PRIVATE ./input_files PRIVATE ./output_files PRIVATE ./pose)
