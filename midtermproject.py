#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 08:06:03 2018

@author: dillon
"""

import numpy as np
#import cv2 as cv
from scipy import misc
import imageio
import glob
import os
# import keras for the convultional stuff
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten,Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import matplotlib.pyplot as plt
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2

# loads all the files from a folder and then returns them as a numpy array
# https://stackoverflow.com/questions/31386096/importing-png-files-into-numpy
# https://stackoverflow.com/questions/31386096/importing-png-files-into-numpy/47044303#47044303
def loadfolder(path):
    # count the number of files in the folder 
    filecount = 0  
    for fn in os.listdir(path):
        filecount += 1
    
    index = 0 # index value for file names
    emptyshape = (filecount,196608) # length of a 1d array of stuff for this project
    imgarray = np.empty(emptyshape,dtype=np.int32)
    for image_path in glob.glob(path+"/*.png"):
        image = imageio.imread(image_path)
        flatimage = image.flatten() # make than image 1 dementional 
        imgarray[index] = flatimage
        index += 1
        
    return imgarray

# lenet
def lenet(width, height, depth, classes):
    pass

# alexnet
# usefull links:
# https://www.mydatahack.com/building-alexnet-with-keras/
def alexnet():
    model = Sequential()
    # first convultional layer
    model.add(Conv2D(filters=96, input_shape=(224,224,3), kernel_size=(11,11),strides=(4,4), padding='valid'))
    model.add(Activation('relu'))
    # pooling
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='valid'))
    model.add(BatchNormalization()) # batch add normalisation
    # second convolutional layer 
    model.add(Conv2D(filters=256,kernel_size=(11,11),strides=(1,1),padding='valid'))
    model.add(Activation('relu'))
    # pooling
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='valid'))
    model.add(BatchNormalization())
    # 3erd convolutional layer
    model.add(Conv2D(filters=384,kernel_size=(3,3),strides=(1,1),padding='valid'))
    model.add(Activation('relu'))
    # batch normalisation
    model.add(BatchNormalization())
    # 4th convolutional layer 
    model.add(Conv2D(filters=384,kernel_size=(3,3),strides=(1,1),padding='valid'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    # 5th convolutional layer 
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    # pooling layer 
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='valid'))
    model.add(BatchNormalization())
    # passing to a dence layer
    model.add(Flatten())
    # 1st dence layer
    model.add(Dense(4096,input_shape=(224*224*3,)))
    model.add(Activation('relu'))
    # Add dropout to prevent overfitting
    model.add(Dropout(0.4))
    model.add(BatchNormalization())
    # 2ed dence layer
    model.add(Dense(4096))
    model.add(Activation('relu'))
    # add dropout
    model.add(Dropout(0.4))
    model.add(BatchNormalization())
    # 3erd dence layer
    model.add(Dense(1000))
    model.add(Activation('relu'))
    # add another dropout
    model.add(Dropout(0.4))
    model.add(BatchNormalization())
    # output layer
    model.add(Dense(17))
    model.add(Activation('softmax'))
    return model

# download all the contents of a folder into a numpy matrix
# define the paths ...
#imgpath = "./../Midterm_Project_Data" # reltive link to the image files 
imgpath = "./Midterm_Project_Data"
testdata_path = imgpath + "/test"
trainingdata_path = imgpath + "/train"
background = "/background"
mal = "/mal"
ben = "/ben"
testben = testdata_path + ben
testmal = testdata_path + mal
testbackground = testdata_path + background
# paths for training data 
trainben = trainingdata_path + mal
trainmal = trainingdata_path + ben
trainbackground = trainingdata_path + background

# load all the images from their respective file
print("loading test and training images...")
test_benign = loadfolder(testben)
train_benign = loadfolder(trainben)
test_malig = loadfolder(testmal)
train_malig = loadfolder(trainmal)
train_background = loadfolder(trainbackground)
test_background = loadfolder(testbackground)
print("done!")
print("now training network...")
alexnet = alexnet() # define an instance of an alexnet 
#alexnet.summary() # prints out a summery of the network
# compile the network
alexnet.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])