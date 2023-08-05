//
// Created by raviprabhashankar on 03.08.23.
//

#ifndef ATHLETEDETECTIONPROJECT_COMMONS_H
#define ATHLETEDETECTIONPROJECT_COMMONS_H

#include <iostream>
#include <fstream>
#include <map>

#define LINUX
#define MPI

namespace attributes_common
{
    #ifdef LINUX

        /* The function returns a modified string with all the characters except the ones appearing beyond the last occurrence
             * of the hook character (including the hook character)
             */
        std::string extractRelevantName(const std::string &input_string, const std::string &hook_char = "/") {
            std::size_t position = input_string.find_last_of(hook_char);
            return input_string.substr(0, position);
        }

    #endif

    #ifdef WINDOWS
        std::string extractRelevantName(const std::string& input_string, const std::string& hook_char = "\\")
                {
                    std::size_t position = input_string.find_last_of(hook_char);
                    return input_string.substr(0, position);
                }
    #endif

    void isFilepathPresent(const std::string& file_path);
    std::string selectVideoFile();
}

namespace attributes_pose_estimation
{
    #ifdef MPI
        const int POSE_PAIRS[14][2] = {
                {0,  1},
                {1,  2},
                {2,  3},
                {3,  4},
                {1,  5},
                {5,  6},
                {6,  7},
                {1,  14},
                {14, 8},
                {8,  9},
                {9,  10},
                {14, 11},
                {11, 12},
                {12, 13}
        };

        std::string proto_file =
                attributes_common::extractRelevantName(__FILE__) + "/pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt";
        std::string weights_file =
                attributes_common::extractRelevantName(__FILE__) + "/pose/mpi/pose_iter_160000.caffemodel";

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

        std::string proto_file =
                attributes_common::extractRelevantName(__FILE__) + "/pose/coco/pose_deploy_linevec.prototxt";
        std::string weights_file =
                attributes_common::extractRelevantName(__FILE__) + "/pose/coco/pose_iter_440000.caffemodel";

        int n_points = 18;
    #endif
}

namespace attributes_object_detection
{
    std::vector<std::string> loadClassNames(const std::string& class_list_filepath);
}

void attributes_common::isFilepathPresent(const std::string& file_path)
{
    std::ifstream file(file_path);
    if(!file.is_open())
    {
        std::cerr << "Filepath " << file_path << " not present. Terminating!";
        exit(EXIT_FAILURE);
    }
    else
    {
        std::cout << "Filepath " << file_path << ": CORRECT!" << std::endl;
    }
}


//========================= FUNCTION DEFINITIONS ==========================================
std::string attributes_common::selectVideoFile()
{
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
                video_file = available_videos.at(choice);
                std::cout   << "Choice: " << choice << " with value: "
                            << attributes_common::extractRelevantName(__FILE__)
                            << "/input_files/"
                            << video_file << std::endl;
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

    return video_file;
}

std::vector<std::string> attributes_object_detection::loadClassNames(const std::string& class_list_filepath)
{
    attributes_common::isFilepathPresent(class_list_filepath);
    std::ifstream file(class_list_filepath);

    std::string current_line;
    std::vector<std::string> class_names;
    while(getline(file, current_line))
    {
        class_names.push_back(current_line);
    }

    return class_names;
}

#endif //ATHLETEDETECTIONPROJECT_COMMONS_H
