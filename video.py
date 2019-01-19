import openface
import cv2
import dlib

import argparse
import os
import numpy as np 

import imutils

from face_utils import Face_utils

import pandas as pd





fileDir = os.path.expanduser("~/openface")
modelDir = os.path.join(fileDir, 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')



parser = argparse.ArgumentParser()

# parser.add_argument('imgs', type=str, nargs='+', help="Input images.")
parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.",
                    default=os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
parser.add_argument('--networkModel', type=str, help="Path to Torch network model.",
                    default=os.path.join(openfaceModelDir, 'nn4.small2.v1.t7'))
parser.add_argument('--imgDim', type=int,
                    help="Default image dimension.", default=96)

parser.add_argument('-v','--vid',type = str, help = "Path to video file")

parser.add_argument('--verbose', action='store_true')


args = parser.parse_args()






### database 
face_database = pd.read_csv('database.csv',index_col = "Name")
face_utils = Face_utils(fileDir, modelDir, dlibModelDir, openfaceModelDir)
##### database end 
align = openface.AlignDlib(args.dlibFacePredictor)
net = openface.TorchNeuralNet(args.networkModel)
detector = dlib.get_frontal_face_detector()

cap = cv2.VideoCapture(args.vid)


while(True):
    # Capture frame-by-frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=500)
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray,1)

    bbs = align.getAllFaceBoundingBoxes(frame)
    
    for (i, bb) in enumerate(bbs):
        #cv2.imshow("frame",rect)
        #embedding = embedding(rect)
        alignedFace = align.align(96, frame, bb,
                                      landmarkIndices=openface.AlignDlib.INNER_EYES_AND_BOTTOM_LIP)
        embedding = net.forward(alignedFace)
        name = face_utils.who_is_it(face_database,embedding)
        
        (x,y,w,h) = face_utils.rect_to_bb(bb)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.putText(frame, name, (x, y + h), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        
        #out.write(frame)
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    


    
# When everything done, release the capture
cap.release()
# out.release()
cv2.destroyAllWindows()

