
from scale.scale_track import track 
from centroid_Tracking.trackableobject import TrackableObject

from functions.speeds import speeds_Dict
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import pandas as pd
import dlib
import cv2
import os.path
import csv

def sddMobilenet_Scale_Tracker(auxPath, current_path):
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", type=str, default=auxPath,
                    help="path del video de entrada") 
    ap.add_argument("-c", "--confidence", type=float, default=0.1,
                    help="minimo de probabilidad para filtrar las detecciones")
    ap.add_argument("-s", "--skip-frames", type=int, default=2,
                    help="numero a quitar de frames")
    args = vars(ap.parse_args())

    
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
        "sofa", "train", "tvmonitor"]

    
    str_caffe_prototxt = str(current_path) + 'mobilenet_ssd/MobileNetSSD_deploy.prototxt'
    str_caffe_model = str(current_path) + 'mobilenet_ssd/MobileNetSSD_deploy.caffemodel'
    prototxtCaffe = str_caffe_prototxt
    caffeModel = str_caffe_model

    net = cv2.dnn.readNetFromCaffe(prototxtCaffe, caffeModel)
    
    vs = cv2.VideoCapture(args["input"])
    
    diccionarioTotal = {}
    
    c = 0
   
    framesRate = vs.get(cv2.CAP_PROP_FPS)
    numberFrames = vs.get(cv2.CAP_PROP_FRAME_COUNT)
    tiempoVideo = (numberFrames / framesRate)
    
    W = None
    H = None
    
    ct = track(maxDisappeared=40, maxDistance=50)
    trackers = []
    pruebaBoxes = []
    trackableObjects = {}

    totalFrames = 0

    while (vs.isOpened()):
        
        frame = vs.read()
        frame = frame[1] if args.get("input", False) else frame

        if args['input'] is not None and frame is None:
            break

        frame = imutils.resize(frame, width=500)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if W is None or H is None:
            (H, W) = frame.shape[:2]

        rects = []
        trackers = []

        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
        net.setInput(blob)
        detections = net.forward()
        auxlstBoxEachT = []

        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0,0,i,2]

            if confidence > args["confidence"]:
                idx = int(detections[0,0,i,1])

                if CLASSES[idx] != "car":
                    continue

                box = detections[0,0,i,3:7] * np.array([W,H,W,H])
                (startX, startY, endX, endY) = box.astype("int")

                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(int(startX), int(startY), int(endX), int(endY))
                tracker.start_track(rgb, rect)
                tracker.update(rgb)
                pos = tracker.get_position()
                startXX = int(pos.left())
                startYY = int(pos.top())
                endXX = int(pos.right())
                endYY = int(pos.bottom())

                bdgBox = np.array([startXX, startYY, endXX, endYY])

                auxlstBoxEachT.append(bdgBox.astype("int"))
                
        pruebaBoxes.append(auxlstBoxEachT)
    vs.release()
    
    diccionarioTotal = {}
    c=0
    for tt in pruebaBoxes:
        objects, objetosID = ct.update(tt)

    
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
        
    diccionarioTotalVeloc = speeds_Dict(diccionarioTotal, tiempoVideo)
    df_estimated_speeds = pd.DataFrame.from_dict(diccionarioTotalVeloc, orient='index')
    
    return df_estimated_speeds
    


























































