//
// Created by raviprabhashankar on 02.08.23.
//

#ifndef ATHLETEDETECTIONPROJECT_PROCESSINGDATA_H
#define ATHLETEDETECTIONPROJECT_PROCESSINGDATA_H

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

namespace process_video_file
{

    void executeVideoProcessing(int processing_option);

    void detectAthleteWithBoundingBox(
            cv::Mat frame,
            const cv::HOGDescriptor& obj_detection_hog,
            int frame_width, int frame_height, double frame_aspect_ratio);

    cv::Mat estimatePose(cv::Mat frame, cv::dnn::Net pose_net, int frame_width, int frame_height);
}

#endif //ATHLETEDETECTIONPROJECT_PROCESSINGDATA_H
