cmake_minimum_required(VERSION 3.25)
project(AthleteDetectionProject)

set(CMAKE_CXX_STANDARD 14)

set(INPUT_DATA_FILES_PATH ${CMAKE_CURRENT_SOURCE_DIR}/input_files)
file(GLOB FILES "${INPUT_DATA_FILES_PATH}/*.mp4")
if("${FILES}" STREQUAL "")
    message(SEND_ERROR "ERROR: ${INPUT_DATA_FILES_PATH} is empty. Make available at least one image/video file for processing!")
endif ()
foreach (file ${FILES})
    file(COPY ${file} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
endforeach()

find_package(OpenCV 4.2 REQUIRED)


add_executable(AthleteDetectionProject main.cpp)

target_link_libraries(AthleteDetectionProject ${OpenCV_LIBS})
