
from centroid_Tracking.centroid_Tracker import tracker
from functions.speeds import speeds_Dict

import numpy as np
import argparse
import cv2
import pandas as pd

import tensorflow as tf
import tensornets as nets
import csv
import os.path

def yolo_Centroid_Tracker(auxPath, current_path, prefix_name):
    tf.reset_default_graph()
    ap = argparse.ArgumentParser()
    ap.add_argument("-c","--confidence", type=float, default=0.5,
                    help="minimo de probabilidad")
    ap.add_argument("-t", "--threshold", type=float, default=0.3,
                    help="umbral cuando se aplica la supresion no maxima")
    args = vars(ap.parse_args())
    
    str_time = date_whole.strftime('%d') + '_' + date_whole.strftime('%m') + '_' +date_whole.strftime('%Y') + '_' +date_whole.strftime('%H') + '_' +date_whole.strftime('%M') + '_' +date_whole.strftime('%S') + '_' +date_whole.strftime('%f')
    
    ct = tracker()
    diccionarioTotal = {}
    c = 0
    videoEntrada = cv2.VideoCapture(auxPath)
    framesRate = videoEntrada.get(cv2.CAP_PROP_FPS)
    numberFrames = videoEntrada.get(cv2.CAP_PROP_FRAME_COUNT)
    tiempoVideo = (numberFrames/framesRate)
    inputs = tf.placeholder(tf.float32, [None, 416, 416, 3])
    model = nets.YOLOv3COCO(inputs, nets.Darknet19)
    classes = {'2':'car'}
    list_of_classes = [2]

    with tf.Session() as sess:
        sess.run(model.pretrained())
        cap = cv2.VideoCapture(auxPath) 
        (W, H) = (None, None)
        writer = None

        while(cap.isOpened()):
            (grabbed, frame) = cap.read() 
            if not grabbed:
                break

            if W is None or H is None:
                (H, W) = frame.shape[:2]
            img = cv2.resize(frame, (416,416))
            imge = np.array(img).reshape(-1,416,416,3)

            preds = sess.run(model.preds, {inputs:model.preprocess(imge)})
            boxes = model.get_boxes(preds, imge.shape[1:3])
            rects = []
            boxes1 = np.array(boxes)
            for j in list_of_classes:
                count = 0
                if str(j) in classes:
                    lab = classes[str(j)]
                if len(boxes1) != 0:                    
                    for i in range(len(boxes1[j])):
                        box = boxes1[j][i]

                        if boxes1[j][i][4] >= args["confidence"]: 
                            count += 1
                            startX = box[0]
                            startY = box[1]
                            endX = box[2]
                            endY = box[3]
                            
                            rectP = []
                            rectP.append(startX)
                            rectP.append(startY)
                            rectP.append(endX)
                            rectP.append(endY)

                            rects.append(rectP)
                            cv2.rectangle(img, (box[0], box[1]),(box[2], box[3]), (0,255,0),1)                
                    objects, objetosID = ct.update(rects)
                    pruebaDict = dict(objetosID)
                    if c==0:
                        for ky in (pruebaDict):
                            auxList = []
                            auxList.append(pruebaDict[ky])
                            diccionarioTotal.update({ky:auxList})
                    else:
                        for hk in (pruebaDict.keys()):
                            if (hk not in diccionarioTotal.keys()):
                                band = []
                                band.append(pruebaDict[hk])
                                diccionarioTotal.update({hk: band})
                            else:
                                diccionarioTotal[hk].append(pruebaDict[hk])
                    c = c+1
                    for (objectID, centroid) in objects.items():
                        for kys, tems in diccionarioTotal.items():
                            if len(tems) > 1:
                                if objectID == kys:
                                    p = np.array(tems[-2])
                                    q = np.array(tems[-1])
                                    dist = round((np.linalg.norm(q - p)), 2)
                                    tme = round((1/framesRate), 5)
                                    speed = round(dist/tme)
                                    text = "{}".format(int(speed))
                                    cv2.rectangle(img, (centroid[0], centroid[1] - 30), ((centroid[0] + 15), (centroid[1] - 15)), (255, 255, 255), -1)
                                    cv2.putText(img, text, (centroid[0], centroid[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 0, 0), 2)
                                    cv2.circle(img, (centroid[0], centroid[1]), 1, (0, 255, 0), 2)
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                str_current_path = str(current_path) + '/video_analyzed/' + str(prefix_name) + '_' + str(str_time) + '.avi'
                writer = cv2.VideoWriter(str_current_path, fourcc, 30, (img.shape[1], img.shape[0]), True)
            writer.write(img)
    writer.release()
    cap.release()

    diccionarioTotalVeloc = speeds_Dict(diccionarioTotal, tiempoVideo)
    
    df_estimated_speeds = pd.DataFrame.from_dict(diccionarioTotalVeloc, orient='index')

    return df_estimated_speeds


def yolo_Centroid_Tracker_train(auxPath):
    tf.reset_default_graph()
    ap = argparse.ArgumentParser()
    ap.add_argument("-c","--confidence", type=float, default=0.5,
                    help="minimo de probabilidad")
    ap.add_argument("-t", "--threshold", type=float, default=0.3,
                    help="umbral cuando se aplica la supresion no maxima")
    args = vars(ap.parse_args())

    ct = tracker()
    diccionarioTotal = {}
    c = 0
    
    videoEntrada = cv2.VideoCapture(auxPath)
    framesRate = videoEntrada.get(cv2.CAP_PROP_FPS)
    numberFrames = videoEntrada.get(cv2.CAP_PROP_FRAME_COUNT)
    
    df_estimated_speeds = pd.DataFrame()
    if framesRate > 0:
        tiempoVideo = (numberFrames/framesRate)       
        inputs = tf.placeholder(tf.float32, [None, 416, 416, 3])
        model = nets.YOLOv3COCO(inputs, nets.Darknet19)
    
        
        classes = {'2':'car'}
        list_of_classes = [2]
    
        with tf.Session() as sess:
            sess.run(model.pretrained())
            cap = cv2.VideoCapture(auxPath) 
            (W, H) = (None, None)
            writer = None
    
            while(cap.isOpened()):
                (grabbed, frame) = cap.read() 
                if not grabbed:
                    break
    
                if W is None or H is None:
                    (H, W) = frame.shape[:2]
                img = cv2.resize(frame, (416,416))
                imge = np.array(img).reshape(-1,416,416,3)
    
                preds = sess.run(model.preds, {inputs:model.preprocess(imge)})
                boxes = model.get_boxes(preds, imge.shape[1:3])
                rects = []    
                boxes1 = np.array(boxes)
                for j in list_of_classes:
                    count = 0
                    if str(j) in classes:
                        lab = classes[str(j)]
                    if len(boxes1) != 0:                    
                        for i in range(len(boxes1[j])):
                            box = boxes1[j][i]
    
                            if boxes1[j][i][4] >= args["confidence"]: 
                                count += 1
                                startX = box[0]
                                startY = box[1]
                                endX = box[2]
                                endY = box[3]
                                
                                rectP = []
                                rectP.append(startX)
                                rectP.append(startY)
                                rectP.append(endX)
                                rectP.append(endY)
    
                                rects.append(rectP)
                                       
                        objects, objetosID = ct.update(rects) 
                        pruebaDict = dict(objetosID)
                        if c==0:
                            for ky in (pruebaDict):
                                auxList = []
                                auxList.append(pruebaDict[ky])
                                diccionarioTotal.update({ky:auxList})
                        else:
                            for hk in (pruebaDict.keys()):
                                if (hk not in diccionarioTotal.keys()):
                                    band = []
                                    band.append(pruebaDict[hk])
                                    diccionarioTotal.update({hk: band})
                                else:
                                    diccionarioTotal[hk].append(pruebaDict[hk])
                        c = c+1
                        
        cap.release()    
        
        diccionarioTotalVeloc = speeds_Dict(diccionarioTotal, tiempoVideo)
        df_estimated_speeds = pd.DataFrame.from_dict(diccionarioTotalVeloc, orient='index')
            
    return df_estimated_speeds




