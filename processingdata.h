//
// Created by raviprabhashankar on 02.08.23.
//

#ifndef ATHLETEDETECTIONPROJECT_PROCESSINGDATA_H
#define ATHLETEDETECTIONPROJECT_PROCESSINGDATA_H

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>


void objectDetectionAndBoundingBox(cv::Mat input_image);
void executePoseEstimation();

#endif //ATHLETEDETECTIONPROJECT_PROCESSINGDATA_H
