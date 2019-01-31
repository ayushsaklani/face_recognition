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

parser.add_argument('-ti','--temp_img',type = str, help = "path to temp image folder ",
                    default = '.temp')

parser.add_argument('--verbose', action='store_true')


args = parser.parse_args()






### database 
face_database = pd.read_csv('.database.csv',index_col = "Name")
face_utils = Face_utils(fileDir, modelDir, dlibModelDir, openfaceModelDir)
##### database end 
align = openface.AlignDlib(args.dlibFacePredictor)
net = openface.TorchNeuralNet(args.networkModel)
detector = dlib.get_frontal_face_detector()


face_utils.video_to_images(args.vid)

images = len(os.listdir(os.path.join(args.temp_img,'images')))
font = cv2.FONT_HERSHEY_SIMPLEX
# print("no of images {}",format(images))
for f_no in range(0,images):
    frame = cv2.imread(os.path.join(os.path.join(args.temp_img,'images'),'frame'+str(f_no)+'.jpg'))
    print(f_no)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray,1)

    bbs = align.getAllFaceBoundingBoxes(frame)
    
    if not bbs:
        cv2.imwrite(os.path.join(os.path.join(args.temp_img,'worked'),'frame'+str(f_no)+'.jpg'),frame)   
    else:
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
            
            cv2.imwrite(os.path.join(os.path.join(args.temp_img,'worked'),'frame'+str(f_no)+'.jpg'),frame)
            


face_utils.images_to_video()

cv2.destroyAllWindows()