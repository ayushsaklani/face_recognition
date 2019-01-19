import openface
import cv2
import dlib

import argparse
import os
import numpy as np 
import pandas as pd
from face_utils import Face_utils
import imutils


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
parser.add_argument('--verbose', action='store_true')

parser.add_argument('--path',type = str, help = "Path to first image folder",required = True)


args = parser.parse_args()


face_utils = Face_utils(fileDir, modelDir, dlibModelDir, openfaceModelDir)

path_data = args.path
print("generating database")
img_list = os.listdir(path_data)
print(img_list)
face_database = {}
for i in img_list:
    input_img = face_utils.img_load(i, path = path_data)
    input_img = imutils.resize(input_img,width = 500)
    face_database[i.replace('.jpg', '')] = face_utils.embedding(input_img)
database = pd.DataFrame.from_dict(data= face_database,orient = 'index')
database.index.name = "Name"
database.to_csv('.database.csv')





