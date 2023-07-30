#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <exception>

#define LINUX

#ifdef LINUX
std::string GetCurrentWorkingDir() {
    std::string source_file = __FILE__;

    const char probe = '/';
    std::size_t position = source_file.find_last_of(probe);
    return source_file.substr(0, position);
}
#endif

#ifdef WINDOWS
std::string GetCurrentWorkingDir() {
    std::string source_file = __FILE__;

    const char probe = '/';
    std::size_t position = source_file.find_last_of(probe);
    return source_file.substr(0, position);
#endif



#define COCO

#ifdef MPI
const int POSE_PAIRS[14][2] =   {
                                {0, 1},
                                {1, 2}, {2,3}, {3,4},
                                {1,5}, {5,6},{6,7},
                                {1,14},
                                {14,8}, {8,9},{9,10},
                                {14,11}, {11,12}, {12,13}
                                };

std::string proto_file = "pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt";
std::string weights_file = "pose/mpi/pose_iter_160000.caffemodel";

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

std::string proto_file = "pose/coco/pose_deploy_linevec.prototxt";
std::string weights_file = "pose/coco/pose_iter_440000.caffemodel";

int n_points = 18;
#endif

int main(int argc, char* argv[]) {
    std::cout << "Hello, World!, the OpenCV version is: " << cv::getVersionString() << std::endl;

    std::unordered_map<int, const std::string> available_videos =
            {
                    {1, GetCurrentWorkingDir()+"/input_files/Rafael Nadal Melbourne 2022.mp4"},
                    {2, GetCurrentWorkingDir()+"/input_files/Usain Bolt Lightspeed.mp4"},
                    {3, GetCurrentWorkingDir()+"/input_files/Zaheer Khan Bowling Action.mp4"}
            };

    int choice = 0;
    std::string scanned_value;
    std::string video_file;
    while(choice == 0)
    {
        std::cout   << "Choose one of the following video file for processing: \n"
                    << "\t[1] Tennis: Rafael Nadal in action during Melbourne\n"
                    << "\t[2] Athletics: Usain Bolt's fastest run\n"
                    << "\t[3] Cricket: Zaheer Khan's bowling action\n"
                    << "\tYour choice [1/2/3]: ";

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
                video_file = available_videos.at(choice);
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

    int in_width = 368;
    int in_height = 368;
    double threshold = 0.01;

    cv::VideoCapture video(video_file);
    if(!video.isOpened())
    {
        std::cerr << "Unable to open the file \"" << video_file << "\"" << std::endl;
        return -1;
    }



    return 0;
}
