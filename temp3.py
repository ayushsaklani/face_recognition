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

parser.add_argument('-i','--img',type = str, help = "Path to image file")

parser.add_argument('-o','--out',type = str, help = "Path to output where you want the output")

parser.add_argument('--verbose', action='store_true')


args = parser.parse_args()






### database 
face_database = pd.read_csv('.database.csv',index_col = "Name")
face_utils = Face_utils(fileDir, modelDir, dlibModelDir, openfaceModelDir)
##### database end 
align = openface.AlignDlib(args.dlibFacePredictor)
net = openface.TorchNeuralNet(args.networkModel)
detector = dlib.get_frontal_face_detector()

# load image
frame = cv2.imread(args.img)

font = cv2.FONT_HERSHEY_SIMPLEX

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
rects = detector(gray,1)

bbs = align.getAllFaceBoundingBoxes(frame)
if not bbs:
    print("empty")
else:
    print(bbs)

    

