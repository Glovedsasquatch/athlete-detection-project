#include <iostream>
#include <vector>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#define LINUX

#ifdef LINUX
/* The function returns a modified string with all the characters except the ones appearing beyond the last occurrence
 * of the hook character (including the hook character)
 */
std::string extractRelevantName(const std::string& input_string, const char& hook_char = '/')
{
    std::size_t position = input_string.find_last_of(hook_char);
    return input_string.substr(0, position);
}
#endif

#ifdef WINDOWS
std::string GetCurrentWorkingDir() {
    std::string source_file = __FILE__;

    const char probe = '/';
    std::size_t position = source_file.find_last_of(probe);
    return source_file.substr(0, position);
#endif



#define MPI

#ifdef MPI
const int POSE_PAIRS[14][2] =   {
                                {0, 1},
                                {1, 2}, {2,3}, {3,4},
                                {1,5}, {5,6},{6,7},
                                {1,14},
                                {14,8}, {8,9},{9,10},
                                {14,11}, {11,12}, {12,13}
                                };

std::string proto_file = extractRelevantName(__FILE__) + "/pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt";
std::string weights_file = extractRelevantName(__FILE__) + "/pose/mpi/pose_iter_160000.caffemodel";

int n_points = 15;
#endif

#ifdef COCO
const int POSE_PAIRS[17][2] =   {
                                    {1,2}, {1,5}, {2,3},
                                    {3,4}, {5,6}, {6,7},
                                    {1,8}, {8,9}, {9,10},
                                    {1,11}, {11,12}, {12,13},
                                    {1,0}, {0,14},
                                    {14,16}, {0,15}, {15,17}
                                };

std::string proto_file = extractRelevantName(__FILE__) + "/pose/coco/pose_deploy_linevec.prototxt";
std::string weights_file = extractRelevantName(__FILE__) + "/pose/coco/pose_iter_440000.caffemodel";

int n_points = 18;
#endif

int main(int argc, char* argv[]) {
    std::cout << "Hello, World!, the OpenCV version is: " << cv::getVersionString() << std::endl;

    std::string current_working_directory = extractRelevantName(__FILE__);

    std::map<int, const std::string> available_videos =
            {
                    {1, "01 Ivanova Borislava Aerobics 2022.mp4"},
                    {2, "02 Nadal Practice servicing.mp4"},
                    {3, "03 Usain Bolt Lightspeed.mp4"},
                    {4, "04 Zaheer Khan Bowling Action.mp4"},
                    {5, "05 Katerina Stefanidi pole vault.mp4"},
                    {6, "06 Jang Mi-Ran Weighlifting Beijing Olympics 2008.mp4"},
                    {7, "07 Usain Bolt Sprinting.mp4"}
            };

    int choice = 0;
    std::string scanned_value;
    std::string video_file;
    while(choice == 0)
    {
        std::cout   << "Choose one of the following video input file for processing: \n";
        for(const auto& itr : available_videos)
        {
            std::cout << "\t" << itr.second << "\n";
        }
        std::cout << "\tYour choice [1-" << available_videos.size() <<"]: ";

        std::cin >> scanned_value;
        try 
        {
            choice = std::stoi(scanned_value);

            if(available_videos.find(choice) == available_videos.end())
            {
                choice = 0;
                scanned_value.clear();
                continue;
            }
            else
            {
                video_file = current_working_directory + "/input_files/" + available_videos.at(choice);
                std::cout << "Choice: " << choice << " with value: " << video_file << std::endl;
                break;
            }
        }
        catch (...)
        {
            std::cout << "Invalid choice. Choose 1, 2, or 3!" << std::endl;
            scanned_value.clear();
            continue;
        }
    }

    std::string output_file = extractRelevantName(available_videos[choice], '.');

    int in_width = 368;
    int in_height = 368;
    double threshold = 0.15;

    cv::VideoCapture cap(video_file);
    if(!cap.isOpened())
    {
        std::cerr << "Unable to open the file \"" << video_file << "\"" << std::endl;
        return -1;
    }

    cv::Mat frame, frame_copy;
    int frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);

    cv::VideoWriter video(current_working_directory + "/output_files/" + output_file + "_Output-skeleton.avi",
                          cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                          10,
                          cv::Size(frame_width, frame_height));

    cv::dnn::Net net = cv::dnn::readNetFromCaffe(proto_file, weights_file);
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

        std::vector<cv::Point> points(n_points);
        for(int n = 0; n < n_points; n++)
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

        int n_pairs = sizeof(POSE_PAIRS)/sizeof(POSE_PAIRS[0]);

        for(int n = 0; n < n_pairs; n++)
        {
            cv::Point2f part_A = points[POSE_PAIRS[n][0]];
            cv::Point2f part_B = points[POSE_PAIRS[n][1]];

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


    return 0;
}
