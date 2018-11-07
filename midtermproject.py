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



# this function takes all the data in a folder and generates a numpy array 
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