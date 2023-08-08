//
// Created by raviprabhashankar on 02.08.23.
//

#include <algorithm>
#include <vector>
#include <fstream>
#include "commons.h"
#include "processingdata.h"

std::string current_dir;
std::string input_filepath;
std::string output_filepath;

int FRAME_WIDTH;
int FRAME_HEIGHT;
double FRAME_ASPECT_RATIO;

static std::vector<cv::Rect> found, found_filtered, prev_frame_found_filtered;

void process_video_file::executeVideoProcessing(int video_processing_option)
{
    std::cout << "Hello, World!, the OpenCV version is: " << cv::getVersionString() << std::endl;

    //Selecting input and output video filepath
    current_dir = attributes_common::extractRelevantName(__FILE__);
    input_filepath = attributes_common::selectVideoFile();
    output_filepath = attributes_common::extractRelevantName(input_filepath, ".");
    input_filepath = current_dir + "/input_files/" + input_filepath;

    std::ifstream FILE;
    FILE.open(input_filepath);
    if(!FILE.is_open())
    {
        std::cerr << "Input video file \"" << input_filepath << "\" not attributes_object_detection::found. Terminating!" << std::endl;
        exit(EXIT_FAILURE);
    }
    FILE.close();

    std::string temp_string = output_filepath;
    output_filepath = current_dir + "/output_files/";
    FILE.open(output_filepath);
    if(!FILE.is_open())
    {
        std::cerr << "Output filepath is missing. Ensure the directory \"output_files\" is present in the same folder as main.cpp!" << std::endl;
        exit(EXIT_FAILURE);
    }
    FILE.close();
    output_filepath = output_filepath + temp_string;

    cv::VideoCapture cap(input_filepath);
    if(!cap.isOpened())
    {
        std::cerr << "Unable to open the input video file \"" << input_filepath << "\". Possibly error in file. Terminating!" << std::endl;
        exit(EXIT_FAILURE);
    }

    FRAME_WIDTH = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    FRAME_HEIGHT = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    FRAME_ASPECT_RATIO = ((double)FRAME_WIDTH)/FRAME_HEIGHT;

    if(video_processing_option == 1)
    {
        output_filepath = output_filepath + "_pose_estimation" + attributes_pose_estimation::scheme + ".avi";
    }
    else if(video_processing_option == 2)
    {
        output_filepath = output_filepath + "_bounding_box.avi";
        attributes_object_detection::setParams(FRAME_ASPECT_RATIO);
    }
    else
    {
        output_filepath = output_filepath + "_pose_estimation_" + attributes_pose_estimation::scheme + "_bounding_box.avi";
        attributes_object_detection::setParams(FRAME_ASPECT_RATIO);
    }

    cv::Mat frame, output_frame;
    cv::VideoWriter video(output_filepath,
                          cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                          10,
                          cv::Size(FRAME_WIDTH, FRAME_HEIGHT));

    // Configuring pose-estimation neural network objects
    cv::dnn::Net pose_net = cv::dnn::readNetFromCaffe(attributes_pose_estimation::proto_file,
                                                      attributes_pose_estimation::weights_file);
    pose_net.setPreferableBackend(cv::dnn::DNN_TARGET_CPU);

    // Configuring object-detection histogram-of-gradient objects
    cv::HOGDescriptor obj_detection_hog;
    obj_detection_hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());

    if(video_processing_option == 1)
    {
        while (cv::waitKey(1) < 0)
        {
            cap >> frame;
            if(frame.empty())
            {
                std::cout << "Video stream is over. Terminating!" << std::endl;
                break;
            }

            output_frame = estimatePose(frame, pose_net, FRAME_WIDTH, FRAME_HEIGHT);

            cv::imshow("Output Video", output_frame);
            video.write(output_frame);
        }
    }
    else if(video_processing_option == 2)
    {
        while (cv::waitKey(1) < 0)
        {
            cap >> frame;
            if(frame.empty())
            {
                std::cout << "Video stream is over. Terminating!" << std::endl;
                break;
            }

            detectAthleteWithBoundingBox(frame, obj_detection_hog, FRAME_WIDTH, FRAME_HEIGHT, FRAME_ASPECT_RATIO);
            output_frame = frame.clone();

            for(auto& index : found_filtered)
            {
                if( index.width < attributes_object_detection::WIDTH_SF*FRAME_WIDTH ||
                    index.area() < attributes_object_detection::AREA_SF*FRAME_WIDTH*FRAME_HEIGHT)
                {
                    continue;
                }

                cv::Rect r = index;
                r.x += cvRound(r.width*0.1);
                r.width = cvRound(r.width*0.8);
                r.y += cvRound(r.height*0.07);
                r.height = cvRound(r.height*0.8);
                rectangle(output_frame, r.tl(), r.br(), cv::Scalar(0, 255, 0), 2);
            }

            cv::imshow("Output Video", output_frame);
            video.write(output_frame);
        }
    }

    cap.release();
    video.release();
}





cv::Mat process_video_file::estimatePose(cv::Mat frame, cv::dnn::Net pose_net, int frame_width, int frame_height)
{
    cv::Mat input_blob =
            cv::dnn::blobFromImage(frame, 1.0/255,
                                   cv::Size(attributes_pose_estimation::MODEL_INPUT_WIDTH, attributes_pose_estimation::MODEL_INPUT_HEIGHT),
                                   false, false);

    pose_net.setInput(input_blob);
    cv::Mat output = pose_net.forward();

    int H = output.size[2];
    int W = output.size[3];

    std::vector<cv::Point> points(attributes_pose_estimation::n_points);
    for(int n = 0; n < attributes_pose_estimation::n_points; n++)
    {
        cv::Mat prob_map(H, W, CV_32F, output.ptr(0, n));

        cv::Point2f p(-1, -1);
        cv::Point max_loc;
        double prob;
        cv::minMaxLoc(prob_map, 0, &prob, 0, &max_loc);

        if(prob > attributes_pose_estimation::THRESHOLD)
        {
            p = max_loc;
            p.x *= (float)frame_width / W;
            p.y *= (float)frame_height / H;

            cv::circle(frame, cv::Point((int)p.x, (int)p.y), 8, cv::Scalar(0, 255, 255), -1);
            cv::putText(
                    frame,
                    cv::format("%d", n),
                    cv::Point((int)p.x, (int)p.y),
                    cv::FONT_HERSHEY_COMPLEX,
                    0.7,
                    cv::Scalar(0, 0, 255), 1);
        }
        points[n] = p;
    }

    int n_pairs = sizeof(attributes_pose_estimation::POSE_PAIRS)/sizeof(attributes_pose_estimation::POSE_PAIRS[0]);

    for(int n = 0; n < n_pairs; n++)
    {
        cv::Point2f part_A = points[attributes_pose_estimation::POSE_PAIRS[n][0]];
        cv::Point2f part_B = points[attributes_pose_estimation::POSE_PAIRS[n][1]];

        if(part_A.x <= 0 || part_A.y <= 0 || part_B.x <= 0 || part_B.y <= 0)
            continue;

        cv::line(frame, part_A, part_B, cv::Scalar(0, 255, 255), 2);
        cv::circle(frame, part_A, 2, cv::Scalar(0, 0, 255), -1);
        cv::circle(frame, part_B, 2, cv::Scalar(0, 0, 255), -1);
    }

    return frame;
}





void process_video_file::detectAthleteWithBoundingBox(
        cv::Mat frame, const cv::HOGDescriptor& obj_detection_hog, int frame_width, int frame_height, double frame_aspect_ratio)
{
    if (found_filtered.size() != 0)
    {
        prev_frame_found_filtered.clear(); prev_frame_found_filtered.resize(0);
        prev_frame_found_filtered = found_filtered;
    }

    found.clear(); found.resize(0);
    found_filtered.clear(); found_filtered.resize(0);

    obj_detection_hog.detectMultiScale
        (frame,
         found, 0,
         cv::Size(4, 4),
         cv::Size(attributes_object_detection::WIDTH_PADDING, attributes_object_detection::HEIGHT_PADDING),
         1.1, 2);

    std::size_t i, j;
    for (i = 0; i < found.size(); i++)
    {
        cv::Rect r = found[i];
        for (j = 0; j < found.size(); j++)
        {
            if (j != i && (r & found[j]) == r)
            {
                break;
            }
        }
        if (j == found.size())
        {
            found_filtered.push_back(r);
        }
    }

    if(prev_frame_found_filtered.size() != 0 && found_filtered.size() == 0)
    {
        found_filtered = prev_frame_found_filtered;
    }
}