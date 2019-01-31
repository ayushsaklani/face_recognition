import numpy as np
import cv2
import os
import argparse
import pandas as pd
from face_utils import Face_utils
import openface
import dlib
 


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
face_database = pd.read_csv('.database.csv',index_col = "Name")
face_utils = Face_utils(fileDir, modelDir, dlibModelDir, openfaceModelDir)
##### database end 
align = openface.AlignDlib(args.dlibFacePredictor)
net = openface.TorchNeuralNet(args.networkModel)
detector = dlib.get_frontal_face_detector()


face_utils.images_to_video()