#include <iostream>
#include <vector>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "processingdata.h"



int main(int argc, char* argv[])
{
    cv::Mat input_image = cv::imread("/home/raviprabhashankar/CLionProjects/AthleteDetectionProject/input_files/input_image.png");
    if(input_image.empty())
    {
        std::cerr << "Image file could not be opened!" << std::endl;
        exit(EXIT_FAILURE);
    }

    objectDetectionAndBoundingBox(input_image);
    executePoseEstimation();
}
