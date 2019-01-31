import openface
import cv2
import dlib

import argparse
import os
import shutil
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

    def video_to_images(self,vid,path=".temp"):
        self.image_path = os.path.join(path,'images')
        self.worked_path = os.path.join(path,'worked')
        if os.path.exists(self.image_path) is True:
            shutil.rmtree(self.image_path)
            os.mkdir(self.image_path)
        else:
            os.mkdir(self.image_path)

        if os.path.exists(self.worked_path) is True:
            shutil.rmtree(self.worked_path)
            os.mkdir(self.worked_path)
        else:
            os.mkdir(self.worked_path)
                

        cap = cv2.VideoCapture(vid)
        i= 0
        while(True):
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret == True:
                cv2.imwrite(self.image_path+"/frame"+str(i)+'.jpg',frame)
                i = i+1
                if i%100 == 0:
                    print("Frame"+str(i)+" saved")
            # print("Last Frame () is saved",format(str(i)))
            else:
                break

        # Display the resulting frame
    

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()

    def images_to_video(self ,out_name = 'video.avi'):
        
        images = ["frame"+str(i)+'.jpg' for i in range(len(os.listdir(self.image_path))) ]
        
        frame = cv2.imread(os.path.join(self.image_path, images[0]))
        height, width, layers = frame.shape

        video = cv2.VideoWriter(out_name, 0, 24, (width,height))

        for image in images:
            video.write(cv2.imread(os.path.join(self.worked_path,image)))
           
        cv2.destroyAllWindows()
        video.release()
            ### custom funvtions end
