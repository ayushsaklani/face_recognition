import openface
import cv2
import dlib

import argparse
import os
import numpy as np 


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

parser.add_argument('--img1',type = str, help = "Path to first image",required = True)

parser.add_argument('--img2',type = str, help = "Path to second image", required = True)

args = parser.parse_args()





img1 = cv2.imread(args.img1)
img2 = cv2.imread(args.img2)

# `args` are parsed command-line arguments.

align = openface.AlignDlib(args.dlibFacePredictor)
net = openface.TorchNeuralNet(args.networkModel, args.imgDim)

# `img` is a numpy matrix containing the RGB pixels of the image.
bb = align.getLargestFaceBoundingBox(img1)
alignedFace1 = align.align(args.imgDim, img1, bb,
                          landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
rep1 = net.forward(alignedFace1)

bb2 = align.getLargestFaceBoundingBox(img2)
alignedFace2 = align.align(args.imgDim, img2, bb2,
                          landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
rep2 = net.forward(alignedFace2)



distance = np.dot(rep1-rep2,rep1-rep2)



print("distance is {}".format(distance))




