//
// Created by raviprabhashankar on 02.08.23.
//

#include <algorithm>
#include <vector>
#include "commons.h"
#include "processingdata.h"

//std::string yolo_model = attributes_common::extractRelevantName(__FILE__) + "/pose/yolo/yolov5s.onnx";
const std::string YOLO_VERSION = "v4-tiny";
std::string yolo_model_cfg = attributes_common::extractRelevantName(__FILE__) + "/pose/yolo/yolo" + YOLO_VERSION + ".cfg";
std::string yolo_model_weight = attributes_common::extractRelevantName(__FILE__) + "/pose/yolo/yolo" + YOLO_VERSION + ".weights";
std::string class_list_filepath = attributes_common::extractRelevantName(__FILE__) + "/input_files/classes.txt";

const float INPUT_WIDTH = 640.0;
const float INPUT_HEIGHT = 640.0;
const float SCORE_THRESHOLD = 0.2;
const float NMS_THRESHOLD = 0.4;


struct Detection
{
    int m_class_id;
    float m_confidence;
    cv::Rect m_box;
};




void objectDetectionAndBoundingBox(cv::Mat input_image)
{
    attributes_common::isFilepathPresent(yolo_model_cfg);
    attributes_common::isFilepathPresent(yolo_model_weight);


    auto net = cv::dnn::readNetFromDarknet(yolo_model_cfg, yolo_model_weight);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    auto model = cv::dnn::DetectionModel(net);
    model.setInputParams(1./255, cv::Size(416, 416), cv::Scalar(), true);




    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    model.detect(input_image, classIds, confidences, boxes, .2, .4);

    int detections = (int)classIds.size();
    const std::vector<cv::Scalar> colors = {cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 0)};
    auto class_list = attributes_object_detection::loadClassNames(class_list_filepath);

    for (int i = 0; i < detections; ++i)
    {
        auto box = boxes[i];
        auto classId = classIds[i];
        const auto color = colors[classId % colors.size()];
        cv::rectangle(input_image, box, color, 3);

        cv::rectangle(input_image, cv::Point(box.x, box.y - 20), cv::Point(box.x + box.width, box.y), color, cv::FILLED);
        cv::putText(input_image, class_list[classId], cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }

    cv::imshow("output", input_image);
    cv::waitKey(0);
}







void executePoseEstimation()
{
    std::cout << "Hello, World!, the OpenCV version is: " << cv::getVersionString() << std::endl;

    std::string current_working_directory = attributes_common::extractRelevantName(__FILE__);
    auto video_file = attributes_common::selectVideoFile();
    auto output_file = attributes_common::extractRelevantName(video_file, ".");
    video_file = current_working_directory + "/input_files/" + video_file;

    int in_width = 368;
    int in_height = 368;
    double threshold = 0.15;

    cv::VideoCapture cap(video_file);
    if(!cap.isOpened())
    {
        std::cerr << "Unable to open the file \"" << video_file << "\"" << std::endl;
        exit(EXIT_FAILURE);
    }

    cv::Mat frame, frame_copy;
    int frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);

    cv::VideoWriter video(current_working_directory + "/output_files/" + output_file + "_Output-skeleton.avi",
                          cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                          10,
                          cv::Size(frame_width, frame_height));

    cv::dnn::Net net = cv::dnn::readNetFromCaffe(   attributes_pose_estimation::proto_file,
                                                    attributes_pose_estimation::weights_file);
    net.setPreferableBackend(cv::dnn::DNN_TARGET_CPU);

    double t;
    while(cv::waitKey(1) < 0)
    {
        t = (double)cv::getTickCount();

        cap >> frame;
        frame_copy = frame.clone();
        cv::Mat input_blob = cv::dnn::blobFromImage(
                frame, 1.0/255, cv::Size(in_width, in_height), false, false);


        net.setInput(input_blob);
        cv::Mat output = net.forward();

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

            if(prob > threshold)
            {
                p = max_loc;
                p.x *= (float)frame_width / W;
                p.y *= (float)frame_height / H;

                cv::circle(frame_copy, cv::Point((int)p.x, (int)p.y), 8, cv::Scalar(0, 255, 255), -1);
                cv::putText(
                        frame_copy,
                        cv::format("%d", n),
                        cv::Point((int)p.x, (int)p.y),
                        cv::FONT_HERSHEY_COMPLEX,
                        1.1,
                        cv::Scalar(0, 0, 255), 2);
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

            cv::line(frame, part_A, part_B, cv::Scalar(0, 255, 255), 3);
            cv::circle(frame, part_A, 3, cv::Scalar(0, 0, 255), -1);
            cv::circle(frame, part_B, 3, cv::Scalar(0, 0, 255), -1);
        }

        t = ((double)cv::getTickCount() - t)/cv::getTickFrequency();
        cv::putText(
                frame,
                cv::format("time taken = %.2f sec", t),
                cv::Point(50, 50),
                cv::FONT_HERSHEY_COMPLEX,
                0.8, cv::Scalar(255, 50, 0), 2);

        cv::imshow("Output-Skeleton", frame);
        video.write(frame);
    }

    cap.release();
    video.release();
}


/*cv::Mat objectDetectionAndBoundingBox(cv::Mat input_image)
{
    int col = input_image.cols;
    int row = input_image.rows;
    int max_dim = std::max(col, row);

    cv::Mat resized_image = cv::Mat::zeros(max_dim, max_dim, CV_8UC3);
    input_image.copyTo(resized_image(cv::Rect(0, 0, col, row)));

    auto net = cv::dnn::readNet(yolo_model);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    //resizing, normalizing to [0, 1), and swap the R and B channels
    cv::Mat result;
    int in_width = 640;
    int in_height = 640;
    cv::dnn::blobFromImage(input_image, result,
                           1./255.,
                           cv::Size(in_width, in_height),
                           cv::Scalar(), true, false);

    std::vector<cv::Mat> predictions;
    net.forward(predictions, net.getUnconnectedOutLayersNames());
    const cv::Mat& output = predictions[0];

    // Detection
    float x_factor = (float)input_image.cols / INPUT_WIDTH;
    float y_factor = (float)input_image.rows / INPUT_HEIGHT;

    float* data = (float*)output.data;

    const int dimension = 85;
    const int rows = 25200;

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    std::vector<std::string> class_names = loadClassNames();

    for(int i = 0; i < rows; ++i)
    {
        float confidence = data[4];
        if(confidence >= 0.4)
        {
            float* classes_scores = data + 5;
            cv::Mat scores(1, (int)class_names.size(), CV_32FC1, classes_scores);
            cv::Point class_id;
            double max_class_score;
            cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
            if(max_class_score > SCORE_THRESHOLD)
            {
                confidences.push_back(confidence);
                class_ids.push_back(class_id.x);

                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];
                int left = int((x - 0.5 * w) * x_factor);
                int top = int((y - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);

                cv::Rect rectangle(left, top, width, height);
                boxes.push_back(rectangle);
            }
        }

        data += 85;
    }

    std::vector<int> nms_result;
    std::vector<Detection> final_result;
    cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, nms_result);
    std::cout << "NMS result size: " << nms_result.size() << std::endl;
    for(auto i : nms_result) {
        Detection detection;
        detection.m_class_id = class_ids[i];
        detection.m_confidence = confidences[i];
        detection.m_box = boxes[i];
        final_result.push_back(detection);
    }

    const std::vector<cv::Scalar> colors = {cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 0)};


    for(auto val : final_result)
    {
        auto box = val.m_box;
        auto classId = val.m_class_id;
        const auto color = colors[classId % colors.size()];
        cv::rectangle(input_image, box, color, 3);

        cv::rectangle(input_image, cv::Point(box.x, box.y - 20), cv::Point(box.x + box.width, box.y), color, cv::FILLED);
        cv::putText(input_image, class_names[classId], cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }

    cv::imshow("Object Detection", input_image);
}
*/
