#include <iostream>
#include "processingdata.h"

int main(int argc, char* argv[])
{
    std::string val = "0";
    int video_processing_option = 0;
    while(video_processing_option == 0)
    {
        std::cout   << "Following athlete detection video processing options are available: \n"
                    << "\tOption 1: Pose Estimation\n"
                    << "\tOption 2: Bounding Box\n"
                    << "Choose one of the above [1/2]: ";

        getline(std::cin, val);
        try
        {
            video_processing_option = std::stoi(val);

            if(!(video_processing_option == 1 || video_processing_option == 2))
            {
                video_processing_option = 0;
                std::cout << "Invalid choice. Choose either 1 or 2!" << std::endl;
                val.clear();
                continue;
            }
        }
        catch (...)
        {
            std::cout << "Invalid choice. Choose either 1 or 2!" << std::endl;
            val.clear();
            continue;
        }
    }

    process_video_file::executeVideoProcessing(video_processing_option);
}