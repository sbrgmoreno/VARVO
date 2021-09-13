
import argparse
import os.path
import video_analysis

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--input_video",type=str, help='path to the input video' )#if the user need analize a video different of the video example project
ap.add_argument("-r", "--route",type=str, help='path to the project folder' )#if a user run the project in another directory in other words a different directory of the command promt
ap.add_argument("-m", "--model",type=str, default = None, help='option to create models with own user videos' )
ap.add_argument("-p", "--prefix",type=str, required=True, help='name of the created files' )
ap.add_argument("-o", "--option",type=str, help='option when user needs to create a model' )
args = vars(ap.parse_args())

current_path = os.getcwd()
#case when the user needs to analyze other videos in a different path of the default
if args['input_video'] and args['route'] and args['model'] == None and args['prefix'] and args['option'] == None:
    str_default_video = str(args['input_video']) 
    if os.path.isfile(str_default_video):
        len_listdir_videos = len(os.listdir(str_default_video))
        if len_listdir_videos > 0:
            for f in os.listdir(str_default_video):
                aux_path_videos = os.path.join(str_default_video, f)#path of the videos to analyze
                video_analysis.video_analysis_models_yolo(aux_path_videos, args['prefix'], str(args['route']))
                video_analysis.video_analysis_models_ssd(aux_path_videos, args['prefix'], str(args['route']))
        else:
            print('The current directory does not have videos')
        #video_analysis.video_analysis_models(arg['input_video'], args['prefix'], current_apth)
    else:
        print('Error... Input video path should be an exintent file')

#case when the user needs to analyze the example video or more than the example video in the default directory
#and run in a different directory
if args['input_video'] == None and args['route'] and args['model'] == None and args['prefix'] and args['option'] == None:
    str_default_video = str(args['route']) + '/video'
    if os.path.isfile(str_default_video):
        len_listdir_videos = len(os.listdir(str_default_video))
        if len_listdir_videos > 0:
            for f in os.listdir(str_default_video):
                aux_path_videos = os.path.join(str_default_video, f)#path of the videos to analyze
                video_analysis.video_analysis_models_yolo(aux_path_videos, args['prefix'], str(args['route']))
                video_analysis.video_analysis_models_ssd(aux_path_videos, args['prefix'], str(args['route']))
        else:
            print('The current directory does not have videos')
        #video_analysis.video_analysis_models(arg['input_video'], args['prefix'], current_apth)
    else:
        print('Error... Input video path should be an exintent file')

#case when the user needs to analyze the example video or more than the example video in the default directory
#and run in the same directory of the command promt
if args['input_video'] == None and args['route'] == None and args['model'] == None and args['prefix'] and args['option'] == None:
    str_default_video = current_path + '/video'
    if os.path.isfile(str_default_video):
        len_listdir_videos = len(os.listdir(str_default_video))
        if len_listdir_videos > 0:
            for f in os.listdir(str_default_video):
                aux_path_videos = os.path.join(str_default_video, f)#path of the videos to analyze
                video_analysis.video_analysis_models_yolo(aux_path_videos, args['prefix'], current_path)
                video_analysis.video_analysis_models_ssd(aux_path_videos, args['prefix'], current_path)
        else:
            print('The current directory does not have videos')
        #video_analysis.video_analysis_models(arg['input_video'], args['prefix'], current_apth)
    else:
        print('Error... Input video path should be an exintent file')

#case when the user needs to analyze the other videos in a different directory
#and run in the same directory of the command promt
if args['input_video'] and args['route'] == None and args['model'] == None and args['prefix'] and args['option'] == None:
    str_default_video = str(args['input_video'])
    if os.path.isfile(str_default_video):
        len_listdir_videos = len(os.listdir(str_default_video))
        if len_listdir_videos > 0:
            for f in os.listdir(str_default_video):
                aux_path_videos = os.path.join(str_default_video, f)#path of the videos to analyze
                video_analysis.video_analysis_models_yolo(aux_path_videos, args['prefix'], current_path)
                video_analysis.video_analysis_models_ssd(aux_path_videos, args['prefix'], current_path)
        else:
            print('The current directory does not have videos')
        #video_analysis.video_analysis_models(arg['input_video'], args['prefix'], current_apth)
    else:
        print('Error... Input video path should be an exintent file')

#case when the user needs to create a model with his data
#and run in the same directory of the command promt
if args['input_video'] == None and args['route'] == None and args['model'] == 'y' and args['prefix'] and args['option']:
    if os.path.isfile(current_path):
        video_analysis.model_creation_user(args['prefix'], current_path, args['option'])
    else:
        print('Error... Current path does not exist')

#case when the user needs to create a model with his data
#and run in other directory of the command promt
if args['input_video'] == None and args['route'] and args['model'] == 'y' and args['prefix'] and args['option']:
    if os.path.isfile(args['route']):
        video_analysis.model_creation_user(args['prefix'], args['route'], args['option'])
    else:
        print('Error... Input video path should be an exintent file')


