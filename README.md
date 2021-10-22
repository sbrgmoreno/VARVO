# VARVO's documentation 
VARVO (Vehicle Accident Recorder from Video Only) implements a novel approach for car crash detection from traffic videos through machine learning techniques. This program was implemented in ``Python`` programming lenguage .

# VARVO analysis workflow

In addition to the expressed above, VARVO generate a video or videos (depending on the case of analysis) with bounding boxes and the estimated speeds of each car found in the video. Another important contribution is about the creation of the own user model of crash detection classification with his / her videos.

**1.** To run VARVO correctly, it must be taken in account that VARVO has five fundamental arguments, these are: input_video, route, model, prefix and option. 
 * *input_video*.- this argument is not mandatory since there is a sample video which could be used to test the algorithm.
 * *route*.- is a non mandatory argument when VARVO is executed in the same location where VARVO is located. In other cases the user must define the path pointing to VARVO.
 * *model*.- this option or argument create models to classify car crashes with other videos (user-provided videos) stored in the VARVO's directories .../crashDetectionProject/user_models/training/incident (normal traffic videos),  and .../crashDetectionProject/user_models/training/no_incident (car crashes videos)
 * *prefix*.- this argument is required to execute VARVO, "prefix" means the user provided name to the analysis
 * *option*.- this argument is required only when the user wants to create his/her own model for classification purposes. In other case this argument is not required. This argument has two values: " 1 " (one) if the user needs a model with YOLO v3 with centroid tracker (feature extractors to obtain estimated speeds) and " 2 " (two) if the user needs a model with SSD Mobilenet with scale tracker (feature extractors to obtain estimated speeds)

**2.** Command prompt argumentsr: 
 * *input_video*.-  ``--v`` or ``--input_video``. Example: ``python ./main_car_speeds.py -v ./videos  -p name_of_analysis`` or ``python ./main_car_speeds.py --input_video ./videos  -p name_of_analysis``
 * *route*.- ``-r`` or ``--route``. Example: ``python ./main_car_speeds.py -r ./Desktop/projects/car_crash_project/main_car_speeds.py  -p name_of_analysis`` or ``python ./main_car_speeds.py --route ./Desktop/projects/car_crash_project/main_car_speeds.py  -p name_of_analysis``
 * *model and option* .- ``-m`` or ``--model``. Example: ``python ./main_car_speeds.py -m y  -p name_of_analysis`` or ``python ./main_car_speeds.py  --model y -p name_of_analysis``, the letter ``y`` means ``yes``
 * *prefix* .- ``-p`` or ``--prefix``. Example: ``python ./main_car_speeds.py -p name_of_analysis`` or ``python ./main_car_speeds.py --prefix name_of_analysis``. Remember this argument is required

**3.** OUTPUT: Once traffic videos are analyzed a four columns  ``CSV`` document is generated:
* *Path_to_video*.- shows the storage path of the processed one-second duration videos.  
* *Elapsed_time_in_seconds*.- shows the timeframe interval that corresponds to each of the one-second processed videos.
* Detected_event.- this column has two possible values, 1 (one) or 0 (zero) depending on whether a car crash was detected or not in the video. 
* *Detected_cars_and_speeds*.- this column has two types of values:
&nbsp;&nbsp; * *nan*.- this type of value means the situation when there are not cars, and
&nbsp;&nbsp; * *integer*.- this type of value is a number that represents the estimated speed of each car in the analyzed.

# Installation
For a comprehensive guide on how to install VARVO and its prerequisites, see [Installation guide](#installation-guide)
# Support
Send additioinal enquiries to mario.moreno01@epn.edu.ec

# Contents
* [VARVO's documentation](#varvos-documentation)
* [Installation guide](#installation-guide)
	* [Prerequisites](#prerequisites)
	* [Installation of VARVO](#installation-guide)
* [VARVO description](#varvo-description)
	* [Directories](#directories)
	* [Files](#files)
	* [Run VARVO](#)
* [Example](#example)

# Installation guide
This installation guide covers three ways to install VARVO, these are: [manually](#manually-installation) (installing the required packages from the command line), [automatic](#automatic-installation) (execute a bash script) and [semi-automatic](#semi-automatic-installation).
## Manual installation 
## Prerequisites
### Fundamental prerequisites  
`` python >= 3.6.10 ``
This item is the unique prerequisite for the installation of VARVO as described below.
### Packages
 * Install following packages using ``pip install``, or if you are working in an anaconda environment using ``conda install``:
	* ``numpy >= 1.17``
	* ``pandas >= 0.25.3``
	* ``opencv >= 4.1.2``
	* ``scikit-learn >= 0.22``
	* ``scipy >= 1.3.2``
	* ``tensorflow >= 1.14.0``
	* ``tensornets >= 0.4.1`` *
	
**Note:** the ``tensornets`` package should be installed only with the command ``pip install``.
## Automatic installation 
 * **Linux**

The automatic installation consist in the execution of the following file: "installer_VARVO.sh" located in the VARVO_bash directory, in the command prompt. Next, when miniconda is installed the command prompt will look lihe: ``(base) command_prompt_path:~$ ``. To complete the installation of the anaconda environment the following command line is essential:
 * ``conda env create -f varvo_linux_env.yml``

Then VARVO program can be executed. For the correct use of VARVO check [VARVO analysis workflow](#varvo-analysis-workflow)

 - **Windows**

In Windows operating system it is easy to onstall all dependencies that VARVO needs. Please, following these steps:

 1. Download and install miniconda through following link: [miniconda](https://docs.conda.io/en/latest/miniconda.html)
2. Open the miniconda command prompt, the prompt will look like: [command prompt example](https://drive.google.com/file/d/18LCJ2KxhbXVd7Lju7O1cbblVKQTj6ltc/view?usp=sharing)
3. Install the VARVO environment with the following line: ``conda env create -f path_of_VARVO_program/environment_VARVO/varvo_linux_env.yml`` or ``conda env create -f path_of_VARVO_program/environment_VARVO/varvo_env.yml``

Then VARVO program can be executed, for the correct use of VARVO check [VARVO analysis workflow](#varvo-analysis-workflow)
## Semi-automatic installation 
This applies when the user has Anaconda installed in the computer, in this case the anaconda environment "varvo_linux_env.yml" should be installed with the following command line:

 -  ``conda env create -f path_of_VARVO_program/environment_VARVO/varvo_linux_env.yml``

## VARVO description
VARVO program contains some directories which fulfill a storage role, this section is a brief description of the VARVO directories and files.

 * #### **Directories**:
	 * *centroid_Tracking*: has one file: 
		 * *centroid_Tacker*:  this file has the functions that compose the centroid tracking.
	 * *data*: has one sub-directory:
		 * *created_video*: saves the cut videos with one second of duration 
	 * *environment_VARVO*: this directory has a file called "VARVO_env.yml" that could be utilized to create an anaconda environment with the all packages that are required to run VARVO program. If exist any problem with this envronment its important to note that the file VARVO_linux_env.yml can be utilized to create the same environment.
	 * *functions*: has one file which saves the necessary functions to compute the distances of the detected cars in the input video.
	 * *mobilenet_ssd*: has two files:
		 * *MobileNetSSD_deploy*.caffemodel: this file saves the weights of the trained network
		 * *MobileNetSSD_deploy*.prototxt: saves the network
	 * *models*: has two sub-directories:
		 * *knn_ssdMob_scale_balanced*: has one file knn_balanced_DLIB_Mobilenet.sav which is the trained model. This model was trained with the  dataset that contains incident an no incidents videos
		 * *knn_yolo_centroid_unbalanced*: has one file knn_unbalanced_centroid_yolo.sav which is the trained model. This model was trained with the a dataset that contains incident an no incidents videos
	 * *scale*: has one file: "scale_track" that has the functions which saves the estimateed speeds
	 * *speeds_analysis*: saves the reports of the analysis of the input video, this file was described in the [section 1](#varvo-analysis-workflow )
	 * *user_models*: has two sub-directories:
		 * *models*: saves the models that users created in two sub-directories: 	
			 * *balanced*: saves the trained model with data balance
			 * *unbalanced*: saves the trained model with imbalanced data
		 * *training*: has three sub-directories:
			 * *cut_videos*: saves the short videos (one second of duration) of incidents and no incidents in the corresponding directories: cut_videos_incident and cut_videos_no_incident 
			 * *incident*: saves user-provided incident videos (car crashes)
			 * *no_incident*: saves user-provided no incident videos (normal traffic)
	 * *video*: has the video or videos that will be analyzed
	 * *video_analyzed*: saves the analyzed video or videos (video with bounding boxes, estimated speeds, etc)
	 * *VARVO_bash*: has one file called "installer_VARVO.sh" that installs <ins> miniconda </ins> and the <ins> anaconda required environment </ins> to run VARVO.
	
 * #### **Files**:
	 * *main_car_speeds*: this file has the principal options that VARVO can execute, for example: create user models with other traffic videos, analyze veideos in the same directory or different to VARVO and other actions expressed above. 
	 * *ssdMobilenet_scale_tracking*: in this file are the principal functions that interact in the object detection with Mobilenet SSD and the tracking detected objects with the scale method.
	 * *yolo_centroid_tracking*: this file contains the principal functions that interact in the object detection with YOLO v3 and the tracking detected objects with the centroid method.
	 * *video_analysis*: this file has the functions that help in the analysis of the input videos either to create models or classify wheter or not is a car crash.
 * #### **Run VARVO**
To run VARVO program in the correct form go to [VARVO analysis workflow](#varvo-analysis-workflow) in the item 3.

## Example
This documentation aims to be a complete example walk through for the usage of VARVO. It assumes you have successfully gone through the [installation](#installation-guide)
### Software specifications
The results provided in the test data were obtained running VARVO with the following software versions.
|O.S:|Windows 10  |
|--|--|
|**Python:**  |v3.6.10 |
|**Python packages:**|``numpy`` v1.17|
||``pandas`` v0.25.3|
||``opencv`` v4.1.2|
||``scikit-learn`` v0.22|
||``scipy`` v1.3.2|
||``tensorflow`` v1.14.0|
||``tensornets`` v0.4.1  |

### Output description
VARVO generates the following output files.
**Note:** file_name represents the prefix that the user enters in the ``-p`` argument, for example:``-p file_name``.

 - *speeds_analysis_unbalanced_file_name.csv* and *speeds_analysis_balanced_file_name.csv*: these files are the reports of the video(s) analysis that have the estimated speeds of the detected cars and the detection of a car crash. The path of these files is: ``./crashDetectionProject/speeds_analysis/`` 

 - *video_%number_%file_name_%day_%month_%year_%hour_%minute_%second_%millisecond*: this file belongs to a set of short videos, in other words is a video of one second of duration.  The character ``%`` represents the value of the word when the video is analized, these values are taken to the system clock, the character ``%number`` is the number of the video. This file path is: ``./crashDetectionProject/data/created_video/``
 - *video_%number_%day_%month_%year_%hour_%minute_%second_%millisecond*: in this file the character ``%`` represents the value of the word when the video is analized, these values are taken to the system clock, the character ``%number`` is the number of the video. This file path is: ``./crashDetectionProject/user_models//training/cut_videos/cut_videos_incident`` or ``./crashDetectionProject/user_models//training/cut_videos/cut_videos_no_incident``
  These videos are generated when a user needs to create new models classification with his / her videos.
  To run correctly VARVO program please check [VARVO analysis workflow](#varvo-analysis-workflow) and [VARVO description](#varvo-description)
### Example
Examples of how to run the differents VARVO options.

 - VARVO is executed in a different path and videos to analize are not in VARVO directory.
 ``./user_path > python ./path_to_VARVO/crashDetectionProject/main_car_speeds.py -p test -r ./path_to_VARVO/crashDetectionProject/ -v ./path_of_videos``
 

 - Run VARVO in a different path and analize the example video or videos in the video VARVO's directory.
 ``./user_path > python ./path_to_VARVO/crashDetectionProject/main_car_speeds.py -p test -r ./path_to_VARVO/crashDetectionProject/``
 

 - VARVO will be execute in the same path and the videos to analize are in the video VARVO's directory.
``./path_to_VARVO/crashDetectionProject > python main_car_speeds.py -p test``

 - VARVO run in the same directory, but the videos to be analized are in other directory to VARVO project.
``./path_to_VARVO/crashDetectionProject > python main_car_speeds.py -p test -v ./path_of_videos``

 - Creation of a user model.
	 - VARVO is executed in the same path and the videos are in the correct directories.
		``./path_to_VARVO/crashDetectionProject > python main_car_speeds.py -p test_model -m y -o 1 `` In this case the option ``-o`` could be ``1`` or ``2``. If is 1 is for YOLO and centroid method and 2 is for Mobilenet SSD with scale method (check [VARVO analysis workflow](#varvo-analysis-workflow) literal 2).
	- VARVO is executed in a different path and the videos are in the correct directories.
		``./path_to_VARVO/crashDetectionProject > python main_car_speeds.py -p test_model -m y -r ./path_to_VARVO/crashDetectionProject/ -o 1 `` In this case the option ``-o`` could be ``1`` or ``2``. If is 1 is for YOLO and centroid method and 2 is for Mobilenet SSD with scale method (check [VARVO analysis workflow](#varvo-analysis-workflow) literal 2).


