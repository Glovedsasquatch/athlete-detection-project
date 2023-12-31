# Athlete Detection Project
The following project aimed at detecting an athlete in every video frame through a 
suitable video processing technique to highlight the detection. In this case, two 
different approaches for object detection, or human detection, have been used, namely,
1. **Bounding box** through histogram of gradients class of OpenCV
2. **Pose Estimation** using datasets obtained from pre-trained models, **MPI**
and **COCO** 

in **OpenCV-C++** framework.

The pre-trained models take input images of a specific size and hence the input frames
have to be scaled down and up (or vice versa) when required.

## A. Bounding box through Histogram-of-Gradients in OpenCV
A very nice idea of how this model works can be found [here](https://learnopencv.com/deep-learning-based-human-pose-estimation-using-opencv-cpp-python/).
The model basically uses gradient field over the pixels to estimate the changes of
pixel values and detecting movement. 
- **cv::HOGDescriptor::getDefaultPeopleDetector()** is used to load the default model
for identification of a person.


## B. Pose Estimation with MPI and COCO Dataset
Pose estimation or keypoint detection is used to detect the skeletal framework of a
person. The object is detected based on the key data points generated by a model 
trained by feeding relevant images to train models (in this case, based on neural 
networks).

- **cv::dnn::forward()** function to parse the image frame to the model leading to
 the prediction of the points
- the key points are used to plot the skeletal framework
- more details on the procedure can be found 
[here](https://learnopencv.com/deep-learning-based-human-pose-estimation-using-opencv-cpp-python/).

> **i. COCO model** produces 18 key data points format
>> - Nose – 0, Neck – 1, Right Shoulder – 2, Right Elbow – 3, 
Right Wrist – 4, Left Shoulder – 5, Left Elbow – 6, Left Wrist – 7, Right Hip – 8, 
Right Knee – 9, Right Ankle – 10, Left Hip – 11, Left Knee – 12, LAnkle – 13, 
Right Eye – 14, Left Eye – 15, Right Ear – 16, Left Ear – 17, Background – 18
> 
> **ii. MPII model** produces 15 key data points format
>> Head – 0, Neck – 1, Right Shoulder – 2, Right Elbow – 3, 
Right Wrist – 4, Left Shoulder – 5, Left Elbow – 6, Left Wrist – 7, Right Hip – 8, 
Right Knee – 9, Right Ankle – 10, Left Hip – 11, Left Knee – 12, Left Ankle – 13, 
Chest – 14, Background – 15 

We are using models trained on Caffe Deep Learning Framework with 2 files:
- **.prototxt** file specifies the architecture of the neutral network
- **.caffemodel** file which stores the weights of the trained model

## Limitations of the methodology
Several limitations were found in the practical excursion to object (human/athlete) 
detection of which some are presented here:
- **Profile bias**: the training image dataset for these object detection models are
humans standing upright and front-facing. Therefore, it is biased against the 
identification of athletes turning backward (for example, in aerobics), flipping (say,
in pole vault), sitting on haunches (for example, in weightlifting), or in a side 
profile, etc. Training the model with a more diversified dataset can yield better 
results.
- **Strong dependency on the image quality**: the models give flawed results 
in poor lighting, lacking contrast, or in specific cases, such as a sprinting athlete,
where the features smudge from one frame into the next. The model would yield a better
result along with better feature extraction algorithms.

## Platform Dependencies and Code Execution
> - Development IDE: **CLion** (works with Visual Studio code but ``current_directory``
> path has to be set manually in the code)
> - OS: **Ubuntu 18.04 (Bionic Beaver)**
> - OpenCV Framework: **version 4.2.0**
> - CMake version: **version 3.25 or higher**
> - File Contents:
>> - ``main.cpp``: inits the video processing technique to be used and call the function 
>> to begin the video processing
>> - ``commons.h``: contains the definitions and parameters for Deep Neural Network schemes
>> other shared parameters, initialization and common functions
>> - ``processingdata.cpp``: implementation of the workflow for reading video files and
>> processing it based on chosen video processing output
>> - Model options can be switched between ``MPI`` to ``COCO`` (or vice versa) in ``commons.h``
>> file (simple change the line ``#define MPI`` to ``#define COCO``)
>> - Operating systems can be changed from ``WINDOWS`` to ``LINUX`` (or vice versa) in a similar 
>> manner (**Caution: the code has not been tested on Windows due to unavailability of 
>> the same**)
>> - ``input_files/``: path to all the input video files
>> - ``pose/``: contains the pre-trained model files
>> - ``output_files/``: path to where all the output files are written to
> - Execution command:
>> 1. Download the following files from the link prescribed:
>> - **Pose-model trained on COCO** from [here](https://www.dropbox.com/s/2h2bv29a130sgrk/pose_iter_440000.caffemodel)
>> and place it in ``./pose/coco/``
>> - **Pose-model trained on MPI** from [here](https://www.dropbox.com/s/drumc6dzllfed16/pose_iter_160000.caffemodel)
>> and place it in ``./pose/mpi/``
>> 2. **for CLion _(recommended)_**: simply open the project in CLion and build and run (or in case of
>> a default keymap, press **Ctrl+Shift+X** to execute)
>> 3. for Visual Studio:
>>- manually set the filepath ``current_dir`` in ``processingdata.cpp``, and ``proto_file``
>>and ``weights_files`` in ``commons.h``; ``current_directory/``: implies the path 
>>containing ``main.cpp``
>>- ``~$ g++ main.cpp processing_data.cpp -o execute_athlete_detection `pkg-config --cflags --libs opencv opencv4` ``
>>- ``~$ ./execute_athlete_detection``

### Additional Comments
- Any video with good contrast, an athlete moving with face directly towards the camera,
would yield the best result. In this case, video ``#03 Zaheer Khan Bowling Action.mp4``
generates the best results.
- Other videos are put to show limitations of the models (although they may show equally 
good results).
- ``CUDA`` acceleration has not been provided because of lack of relevant hardware to
test its features.
