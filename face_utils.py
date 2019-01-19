import openface
import cv2
import dlib

import argparse
import os
import numpy as np 

import imutils



class Face_utils:
    def __init__(self, fileDir, modelDir, dlibModelDir, openfaceModelDir):

        self.fileDir = "/home/root_zero/openface"
        self.modelDir = os.path.join(self.fileDir, 'models')
        self.dlibModelDir = os.path.join(self.modelDir, 'dlib')
        self.openfaceModelDir = os.path.join(self.modelDir, 'openface')
        self.dlibFacePredictor = os.path.join(self.dlibModelDir, "shape_predictor_68_face_landmarks.dat")
        self.networkModel = os.path.join(self.openfaceModelDir, 'nn4.small2.v1.t7')
        self.imgDim = 96




    def rect_to_bb(self,rect):
        # take a bounding predicted by dlib and convert it
        # to the format (x, y, w, h) as we would normally do
        # with OpenCV
        x = rect.left()
        y = rect.top() 
        w = rect.right() - x
        h = rect.bottom()  - y

        # return a tuple of (x, y, w, h)
        return (x, y, w, h)


    #####img _load
    def img_load(self, name, path = "images"):
        img = cv2.imread(os.path.join(path, name), 1)# print(img.shape)
        return img

    ##### embedding
    def embedding(self,img):
        align = openface.AlignDlib(self.dlibFacePredictor)
        net = openface.TorchNeuralNet(self.networkModel, self.imgDim)
        # `img` is a numpy matrix containing the RGB pixels of the image.
        bb = align.getLargestFaceBoundingBox(img)
        alignedFace = align.align(self.imgDim, img, bb,
                              landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        return net.forward(alignedFace)

    ### distance
    def distance(self,embeddings1, embeddings2, distance_metric = 0):
        if distance_metric == 0: #Euclidian distance
            diff = np.subtract(embeddings1, embeddings2)
            dist = np.sum(np.square(diff))
        elif distance_metric == 1: #Distance based on cosine similarity
            dot = np.sum(np.multiply(embeddings1, embeddings2))
            norm = np.linalg.norm(embeddings1) * np.linalg.norm(embeddings2)
            similarity = dot / norm
            dist = np.arccos(similarity) / np.pi
        else :
            raise 'Undefined distance metric %d' % distance_metricr

        return dist


    ### database 
    def database_dict_gen(self):
        path_data = 'images/database'
        print("generating database")
        img_list = os.listdir(path_data)
        print(img_list)
        face_database = {}
        for i in img_list:
            input_img = img_load(i, path = path_data)
            input_img = imutils.resize(input_img,width = 500)
            face_database[i.replace('.jpg', '')] = embedding(input_img)
        return face_database

    #### who _is _it
    # def who_is_it(self,database, embedding):
    #     min_dist = 100
    #     for (name, db_enc) in database.items():
    #         dist = distance(database[name], embedding, distance_metric = 0)
    #         if dist < min_dist:
    #             min_dist = dist
    #             iden = name
    #     if min_dist > 0.9:
    #         identity = "Not in Database"
    #     else :
    #         identity = iden

    #     return identity

    def who_is_it(self,database,embedding):
        min_dist =100
        for name in database.index:
            dist = self.distance(np.array(database.loc[name]),embedding, distance_metric = 0)
            if dist < min_dist:
                min_dist = dist
                iden = name
        if min_dist >0.9:
            identity = "Not in Database"
        else : 
            identity = iden
        return identity

    ### custom funvtions end
