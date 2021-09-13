
import cv2
import numpy as np
import os
import csv
import datetime

from yolo_centroid_tracking import yolo_Centroid_Tracker
from yolo_centroid_tracking import yolo_Centroid_Tracker_train
from ssdMobilenet_scale_tracking import sddMobilenet_Scale_Tracker 


import pandas as pd
import pickle
import statistics

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

#global_path = os.getcwd()

#this function returns or compute the id car, label, speed mean of the cars which had an accident
def label_mean_car_speed (df_estimated_speeds, option, global_path):
    df_estimated_speeds = df_estimated_speeds.fillna(0)
    #option is the flag for the type of the trained model inthis case 
    #1 for knn yolo centroid model
    #2 for knn ssd Mobilenet scale model
    if option == 1:
        file_name_unbalanced_model = str(global_path) + '/models/knn_yolo_centroid_unbalanced/knn_unbalanced_centroid_yolo.sav'#static saved file 
    if option == 2:
        file_name_unbalanced_model = str(global_path) + '/models/knn_ssdMob_scale_balanced/knn_balanced_DLIB_Mobilenet.sav'#static saved file 
        
    loaded_unbalanced_model = pickle.load(open(file_name_unbalanced_model, 'rb'))
    y_pred_unbalanced = loaded_unbalanced_model.predict(df_estimated_speeds)

    #print('Etiquetas: {}'.format(y_pred_unbalanced))#check the labels of the output video ######!!!! not important in the program
    df_y_pred = pd.DataFrame(data=y_pred_unbalanced, columns=['29'])
    df_X_y_pred = pd.concat([df_estimated_speeds, df_y_pred], axis=1)
    cont = 0
    label = 0# zero represents no incident
    car_id = []#accumulator of id cars
    mean_speed = []#accumulator of mean speed of the car id
    for incident in df_X_y_pred['29']:
        cont = cont + 1#the car ID of the detection or the index of the row
        if incident == 1:
            label = 1#one denotes there is a incident qith a car
            mean_car_speed = statistics.mean(df_X_y_pred.iloc[cont-1][:-1])#mean of the car spped
            car_id.append(cont)#car id
            mean_speed.append(mean_car_speed)#mean speeds of car incident
    return car_id, mean_speed, label


#this function returns a DataFrame of the analisys of the videos with 1 second of duration
def incident_analisys_table(video_name, cont_video, car_id, mean_speed, label):
    len_car_id = len(car_id)
    len_mean_speed = len(mean_speed)
    str_veloc = ' '
    if len_car_id == len_mean_speed:
        if len_car_id > 1 and len_mean_speed > 1:
            label = 1
            for l_c_i, l_m_s in zip(car_id, mean_speed):
                str_band = 'Car ' + str(l_c_i) + ' with ' + str(l_m_s) + ', '
                str_veloc = str_veloc + str_band
            speeds = [str_veloc]
        else:
            label = 0
            speeds = ['nan']
        data_analisys = {'Path_to_video' : video_name,
                         'Elapsed_time_in_seconds' : cont_video,
                         'Detected_event' : label,
                         'Detected_cars_and_speeds(km/h)': speeds}
        df_video_analisys = pd.DataFrame(data_analisys, columns=['Path_to_video', 'Elapsed_time_in_seconds', 'Detected_event', 'Detected_cars_and_speeds(km/h)'])
        return df_video_analisys
    else:
        print('Missmatch found between car elements and the number of assigned speeds')
    
#creation of the csv report of the speeds and accident detection
def creation_update_analisys(df_video_analisys, flag_name, option, global_path):
    if option == 1:
        path_csv = str(global_path) + '/speeds_analysis/speeds_analysis_unbalanced_' + flag_name +'.csv'
    if option == 2:
        path_csv = str(global_path) + '/speeds_analysis/speeds_analysis_balanced_' + flag_name +'.csv'

    if os.path.exists(path_csv):
        with open(path_csv, 'a+') as file:
            writerCsv = csv.writer(file, delimiter=',')
            for ind, row in df_video_analisys.iterrows():
                writerCsv.writerow(row)
    else:
        df_video_analisys.to_csv(path_csv, index=None)
        

def video_analysis_models_yolo(video_path, prefix_name, current_path):
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    str_time = date_whole.strftime('%d') + '_' + date_whole.strftime('%m') + '_' +date_whole.strftime('%Y') + '_' +date_whole.strftime('%H') + '_' +date_whole.strftime('%M') + '_' +date_whole.strftime('%S') + '_' +date_whole.strftime('%f')
    width = cap.get(3)
    height = cap.get(4)
    
    str_data_folder = str(current_path) + '/data'
    try :
        if not os.path.exists(str_data_folder):
            os.makedirs(str_data_folder)
    except OSError:
            print('Creating the necessary directories...')
    height, width, layers = frame.shape
    cont_Video = 0
    cont_name = 0
    images = []
    
    while (success):
        #capture frame to frame
        cont_Video = cont_Video + 1#frame counter 
        
        images.append(frame)
        if cont_Video == 30:#if you need more than 1 second please change 30 for (30 * x) where x is the seconds you need
            cont_name = cont_name + 1
            video_name = str_data_folder + '/created_video/output_video_%d_%s_%s.avi'%(cont_name, prefix_name, str_time)
            video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), 30, (width, height))
            for image in images:
                video.write(image)
            
            cv2.destroyAllWindows()
            video.release()
            # ############### YOLO CENTROID MODEL################################################
            path_estimated_speeds_unbalanced = yolo_Centroid_Tracker(video_name, current_path, prefix_name) #returns the estimated speeds with yolo centroid 
            
            if (path_estimated_speeds_unbalanced.empty):
                path_estimated_speeds_unbalanced = pd.DataFrame(np.zeros((1, 29)))
            if path_estimated_speeds_unbalanced.shape[1] < 29:
                flag_row = np.array([])
                acum_rows = []
                path_estimated_speeds_unbalanced = path_estimated_speeds_unbalanced.fillna(0)
                for index, row in path_estimated_speeds_unbalanced.iterrows():
                    row_np = np.array(row)
                    l_row = len(row_np)
                    l_len_rows = 29 - l_row
                    mat_zeros = np.zeros(l_len_rows)
                    mat_total = np.concatenate((row_np, mat_zeros), axis=0)
                    acum_rows.append(mat_total)
                path_estimated_speeds_unbalanced = pd.DataFrame(acum_rows)
            
            car_id, mean_speed,label = label_mean_car_speed(path_estimated_speeds_unbalanced, 1, current_path) #option = 1 because in this case we need the computation with the first model
            df_video_analisys = incident_analisys_table(video_name, cont_name, car_id, mean_speed, label)
            creation_update_analisys(df_video_analisys, prefix_name, 1, current_path)#here!!!!!! shoud be the prefix not the name:'flag1'
            ##########################################################################################
            
            ###########################################################################################
            cont_Video = 0
            images = []
        success, frame = cap.read()
    #the end of the video 
    cap.release()
    cv2.destroyAllWindows()


def video_analysis_models_ssd(video_path, prefix_name, current_path):
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    str_time = date_whole.strftime('%d') + '_' + date_whole.strftime('%m') + '_' +date_whole.strftime('%Y') + '_' +date_whole.strftime('%H') + '_' +date_whole.strftime('%M') + '_' +date_whole.strftime('%S') + '_' +date_whole.strftime('%f')
    width = cap.get(3)
    height = cap.get(4)
    
    str_data_folder = str(current_path) + '/data'
    try :
        if not os.path.exists(str_data_folder):
            os.makedirs(str_data_folder)
    except OSError:
            print('Creating the necessary directories...')
    height, width, layers = frame.shape
    cont_Video = 0
    cont_name = 0
    images = []
    
    while (success):
        #capture frame to frame
        cont_Video = cont_Video + 1#frame counter 
        
        images.append(frame)
        if cont_Video == 30:#if you need more than 1 second please change 30 for (30 * x) where x is the seconds you need
            cont_name = cont_name + 1
            video_name = str_data_folder + '/created_video/output_video_%d_%s_%s.avi'%(cont_name, prefix_name, str_time)
            video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), 30, (width, height))
            for image in images:
                video.write(image)
            
            cv2.destroyAllWindows()
            video.release()
            ##########################################################################################
            ########### SSD MOBILENET SCALE MODEL ###################################################
            
            path_estimated_speeds_balanced = sddMobilenet_Scale_Tracker(video_name, current_path)#returns the estimated speeds with scale ssd Mobilenet
            
            if path_estimated_speeds_balanced.empty:
                path_estimated_speeds_balanced = pd.DataFrame(np.zeros((1, 29)))
            if path_estimated_speeds_balanced.shape[1] < 29:
                flag_row_b = np.array([])
                acum_rows_b = []
                path_estimated_speeds_balanced = path_estimated_speeds_balanced.fillna(0)
                for index_b, row_b in path_estimated_speeds_balanced.iterrows():
                    row_np_b = np.array(row_b)
                    l_row_b = len(row_np_b)
                    l_len_rows_b = 29 - l_row_b
                    mat_zeros_b = np.zeros(l_len_rows_b)
                    mat_total_b = np.concatenate((row_np_b, mat_zeros_b), axis = 0)
                    acum_rows_b.append(mat_total_b)
                path_estimated_speeds_balanced = pd.DataFrame(acum_rows_b)
                
            car_id_b, mean_speed_b, label_b = label_mean_car_speed(path_estimated_speeds_balanced, 2, current_path)#option 2 because in this case we need the analisys with the second model
            df_video_analisys_balanced = incident_analisys_table(video_name, (cont_name + 1), car_id_b, mean_speed_b, label_b)
            creation_update_analisys(df_video_analisys_balanced, prefix_name, 2, current_path)#here!!!!!! shoud be the prefix not the name:'flag2'
            
            ###########################################################################################
            cont_Video = 0
            images = []
        success, frame = cap.read()
    #the end of the video 
    cap.release()
    cv2.destroyAllWindows()




def video_division(video_path_only, folder_save):
    
    date_whole = datetime.datetime.now()
    str_time = date_whole.strftime('%d') + '_' + date_whole.strftime('%m') + '_' +date_whole.strftime('%Y') + '_' +date_whole.strftime('%H') + '_' +date_whole.strftime('%M') + '_' +date_whole.strftime('%S') + '_' +date_whole.strftime('%f')

    cap = cv2.VideoCapture(video_path_only)#video_path is the video
    success, frame = cap.read()
    
    width = cap.get(3)
    height = cap.get(4)
    
    str_data_folder = folder_save#the folder that the videos will be saved in this case '/cut_videos/cut_videos_incident' or '/cut_videos/cut_videos_no_incident'
    
    height, width, layers = frame.shape
    cont_Video = 0
    cont_name = 0
    images = []
    
    while (success):
        #capture frame to frame
        cont_Video = cont_Video + 1#frame counter 
        
        images.append(frame)
        if cont_Video == 30:#if you need more than 1 second please change 30 for (30 * x) where x is the seconds you need
            cont_name = cont_name + 1
            video_name = str_data_folder + '/video_%d_%s.avi'%(cont_name, str_time)
            video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), 30, (width, height))
            for image in images:
                video.write(image)
            
            cv2.destroyAllWindows()
            video.release()
            
            cont_Video = 0
            images = []
        success, frame = cap.read()
    #the end of the video 
    cap.release()
    cv2.destroyAllWindows()
    
def feature_extraction_positive(len_listdir_incident, path_incident_train, option):#ones
    if len_listdir_incident:
        df_data_incident_speeds = pd.DataFrame()
        for f in os.listdir(path_incident_train):
            aux_incident_path = os.path.join(path_incident_train, f)
            
            ###########OPTIONS TO CREATE THE MODEL############
            if option == 1:
                df_estimated_speeds_user = yolo_Centroid_Tracker_train(aux_incident_path)#features of the videos in this case speeds in each frame of the video
            if option == 2:
                df_estimated_speeds_user = sddMobilenet_Scale_Tracker(aux_incident_path, current_path)
            ###################################################
                
            if df_estimated_speeds_user.empty:
                df_estimated_speeds_user = pd.DataFrame(np.zeros((1, 29)))
            if df_estimated_speeds_user.shape[1] < 29:
                flag_row_b = np.array([])
                acum_rows_b = []
                df_estimated_speeds_user = df_estimated_speeds_user.fillna(0)
                for index_b, row_b in df_estimated_speeds_user.iterrows():
                    row_np_b = np.array(row_b)
                    l_row_b = len(row_np_b)
                    l_len_rows_b = 29 - l_row_b
                    mat_zeros_b = np.zeros(l_len_rows_b)
                    mat_total_b = np.concatenate((row_np_b, mat_zeros_b), axis = 0)
                    acum_rows_b.append(mat_total_b)
                df_estimated_speeds_user = pd.DataFrame(acum_rows_b)
            df_data_incident_speeds = df_data_incident_speeds.append(df_estimated_speeds_user)#append the dataFrames of the each video of 1 second
                
                ############!!!!!!!!!AUMENTADO UNA SANGRIA IZQUIERDA <-
        incident_labels = np.ones(df_data_incident_speeds.shape[0])#create ones in the same number of the rows in the dataframe
        ####//####
        print('Incident labels filas: {}'.format( df_data_incident_speeds.shape[0]))########proof dataset
        ####\\####
        df_incident_class = pd.DataFrame(incident_labels, columns=['Class'])#the last column has the name Class because is the label of the speeds arrays
        ######//####
        df_data_incident_speeds = df_data_incident_speeds.fillna(0)
        #df_incident_class.to_csv('path_to_.csv', index=False)##If you need the classes of estimated speeds uncomment this line
        #df_data_incident_speeds.to_csv('path_to_.csv', index=False)##If you need the estimated speeds uncomment this line

        
        df_data_incident_speeds = df_data_incident_speeds.reset_index()
        df_incident_class = df_incident_class.reset_index()
        ######\\####
        df_dataset_incident = pd.concat([df_data_incident_speeds, df_incident_class], axis=1)#, ignore_index=True)#######!!!!this is the total frame of incident !!!!!!!!!!
        df_dataset_incident = df_dataset_incident.drop(['index'], axis=1)
        ######//####
        #df_dataset_incident.to_csv('path_to_.csv', index=False)#If you need the incident estimated speeds uncomment this line
        #####\\####

    else:
        print('there are not videos to analice')
        print('please add videos in the following path: {}'.format(path_incident_train))
    
    return df_dataset_incident

def feature_extraction_negative(len_listdir_no_incident, path_no_incident_train, option):#zeros
    len_listdir_incident = len_listdir_no_incident
    path_incident_train = path_no_incident_train
    
    if len_listdir_incident:
        df_data_incident_speeds = pd.DataFrame()
        for f in os.listdir(path_incident_train):
            aux_incident_path = os.path.join(path_incident_train, f)
            
            ###########OPTIONS TO CREATE THE MODEL############
            if option == 1:
                df_estimated_speeds_user = yolo_Centroid_Tracker_train(aux_incident_path)#features of the videos in this case speeds in each frame of the video
            if option == 2:
                df_estimated_speeds_user = sddMobilenet_Scale_Tracker(aux_incident_path, current_path)
            ###################################################
                
            if df_estimated_speeds_user.empty:
                df_estimated_speeds_user = pd.DataFrame(np.zeros((1, 29)))
            if df_estimated_speeds_user.shape[1] < 29:
                flag_row_b = np.array([])
                acum_rows_b = []
                df_estimated_speeds_user = df_estimated_speeds_user.fillna(0)
                for index_b, row_b in df_estimated_speeds_user.iterrows():
                    row_np_b = np.array(row_b)
                    l_row_b = len(row_np_b)
                    l_len_rows_b = 29 - l_row_b
                    mat_zeros_b = np.zeros(l_len_rows_b)
                    mat_total_b = np.concatenate((row_np_b, mat_zeros_b), axis = 0)
                    acum_rows_b.append(mat_total_b)
                df_estimated_speeds_user = pd.DataFrame(acum_rows_b)
            df_data_incident_speeds = df_data_incident_speeds.append(df_estimated_speeds_user)#append the dataFrames of the each video of 1 second
                
                ############!!!!!!!!!AUMENTADO UNA SANGRIA IZQUIERDA <-
        incident_labels = np.zeros(df_data_incident_speeds.shape[0])#create zeros in the same number of the rows in the dataframe
        ####//####
        print('Incident labels filas: {}'.format( df_data_incident_speeds.shape[0]))########proof dataset
        ####\\####
        df_incident_class = pd.DataFrame(incident_labels, columns=['Class'])#the last column has the name Class because is the label of the speeds arrays
        ######//####
        df_data_incident_speeds = df_data_incident_speeds.fillna(0)
        #df_incident_class.to_csv('/path_to_.csv', index=False)#If you need the classes of estimated speeds uncomment this line
        #df_data_incident_speeds.to_csv('path_to_.csv', index=False)#If you need the estimated speeds uncomment this line

        
        df_data_incident_speeds = df_data_incident_speeds.reset_index()
        df_incident_class = df_incident_class.reset_index()
        ######\\####
        df_dataset_incident = pd.concat([df_data_incident_speeds, df_incident_class], axis=1)#, ignore_index=True)#######!!!!this is the total frame of incident !!!!!!!!!!
        df_dataset_incident = df_dataset_incident.drop(['index'], axis=1)
        ######//####
        #df_dataset_incident.to_csv('/path_to_.csv', index=False)#If you need the no incident estimated speeds uncomment this line
        #####\\####

    else:
        print('there are not videos to analice')
        print('please add videos in the following path: {}'.format(path_incident_train))
    
    return df_dataset_incident
    
def model_creation_user(prefix_name, current_path, option):
    str_usr_modl_folder = str(current_path) + '/user_models'#user models
    try:
        if not os.path.exists(str_usr_modl_folder):
            str_trng_inc = str_usr_modl_folder + '/training/incident'
            os.makedirs(str_trng_inc)
            str_trng_no_inc = str_usr_modl_folder + '/training/no_incident'
            os.makedirs(str_trng_no_inc)
            str_mdl_bal = str_usr_modl_folder + '/models/balanced'
            os.makedirs(str_mdl_bal)
            str_mdl_unb = str_usr_modl_folder + '/models/unbalanced'
            os.makedirs(str_mdl_unb)
            str_mdl_cut_incident = str_usr_modl_folder + '/training/cut_videos/cut_videos_incident'
            os.makedirs(str_mdl_cut_incident)
            str_mdl_cut_no_incident = str_usr_modl_folder + '/training/cut_videos/cut_videos_no_incident'
            os.makedirs(str_mdl_cut_no_incident)
    except OSError:
            print('Creating the necessary directories...')
    
    str_trng_inc = str_usr_modl_folder + '/training/incident'
    str_trng_no_inc = str_usr_modl_folder + '/training/no_incident'
    str_mdl_bal = str_usr_modl_folder + '/models/balanced'
    str_mdl_unb = str_mdl_bal = str_usr_modl_folder + '/models/unbalanced'
    str_mdl_cut_incident = str_usr_modl_folder + '/training/cut_videos/cut_videos_incident'
    str_mdl_cut_no_incident = str_usr_modl_folder + '/training/cut_videos/cut_videos_no_incident'
    
    
    path_brute_incident_videos = str_trng_inc
    path_brute_no_incident_videos = str_trng_no_inc
    len_listdir_incident_brute = len(os.listdir(path_brute_incident_videos))
    len_listdir_no_incident_brute = len(os.listdir(path_brute_no_incident_videos))
    
    if len_listdir_incident_brute:
        for fil in os.listdir(path_brute_incident_videos):
            aux_incident_brute = os.path.join(path_brute_incident_videos, fil)
            video_division(aux_incident_brute, str_mdl_cut_incident)
    else:
        print('there are not videos to analice')
        print('please add videos in the following path: {}'.format(path_brute_incident_videos))
        
        
    if len_listdir_no_incident_brute:
        for fil in os.listdir(path_brute_no_incident_videos):
            aux_no_incident_brute = os.path.join(path_brute_no_incident_videos, fil)
            video_division(aux_no_incident_brute, str_mdl_cut_no_incident)
    else:
        print('there are not videos to analice')
        print('please add videos in the following path: {}'.format(path_brute_no_incident_videos))
    
   
    path_incident_train = str_mdl_cut_incident#'/dataVideosIncidentsVehicle/modelosCreados/crashDetectionProject/user_models/training/incident'
    path_no_incident_train = str_mdl_cut_no_incident#'/dataVideosIncidentsVehicle/modelosCreados/crashDetectionProject/user_models/training/no_incident'
    len_listdir_incident = len(os.listdir(path_incident_train))
    len_listdr_no_incident = len(os.listdir(path_no_incident_train))
    
    df_incident = feature_extraction_positive(len_listdir_incident, path_incident_train, option)
    df_no_incident = feature_extraction_negative(len_listdr_no_incident, path_no_incident_train, option)

    
    df_dataset = df_incident.append(df_no_incident, ignore_index=True, sort=False)#total data to train
    
    ######//####
    #df_dataset.to_csv('/path_to_.csv', index=False)#If you need the estimated speeds uncomment this line
    ####\\####
    
    y = df_dataset['Class']
    X = df_dataset.drop(['Class'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
    knn_unbalanced_model = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None, n_jobs=-1, n_neighbors=9, p=1, weights='distance')
    knn_balanced_model = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None, n_jobs=-1, n_neighbors=9, p=1, weights='distance')
    knn_unbalanced_model.fit(X_train, y_train)
    knn_balanced_model.fit(X_train, y_train)
    unbalanced_file_name = str_usr_modl_folder + '/models/unbalanced/unbalanced_' + str(prefix_name) +'.sav'
    balanced_file_name = str_usr_modl_folder + '/models/balanced/balanced_' + str(prefix_name) +'.sav'
    pickle.dump(knn_unbalanced_model, open(unbalanced_file_name, 'wb'))
    pickle.dump(knn_balanced_model, open(balanced_file_name, 'wb'))











