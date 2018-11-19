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
import pickle
# import keras for the convultional stuff
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten,Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import load_model # added 
import matplotlib.pyplot as plt
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2

# loads all the files from a folder and then returns them as a numpy array
# https://stackoverflow.com/questions/31386096/importing-png-files-into-numpy
# https://stackoverflow.com/questions/31386096/importing-png-files-into-numpy/47044303#47044303
#def loadfolder(path):
#    # count the number of files in the folder 
#    filecount = 0  
#    for fn in os.listdir(path):
#        filecount += 1
#    
#    index = 0 # index value for file names
#    emptyshape = (filecount,196608) # length of a 1d array of stuff for this project
#    imgarray = np.empty(emptyshape,dtype=np.int32)
#    for image_path in glob.glob(path+"/*.png"):
#        image = imageio.imread(image_path)
#        flatimage = image.flatten() # make than image 1 dementional 
#        imgarray[index] = flatimage
#        index += 1
#        
#    return imgarray

def loadfolder(path):
    filecount = len(os.listdir(path)) # counte the number of files in a folder
    # load the images, quick and dirty like
    emptyshape = (filecount,256,256,3)
    data = np.empty(emptyshape)
    index = 0
    for image_path in glob.glob(path+"/*.png"):
        image = imageio.imread(image_path)
        data[index] = image
        index+=1
        
    return data # return the data 
    

# lenet, from Chao's minst hand writing exsample 
def LeNet(width=1, height=1, depth=1, classes=1):
		# initialize the model
		model = Sequential()
		inputShape = (height, width, depth)

		# if we are using "channels first", update the input shape
		#if K.image_data_format() == "channels_first":
		#	inputShape = (depth, height, width)

		# first set of CONV => RELU => POOL layers
		model.add(Conv2D(20, (5, 5), padding="same", input_shape=inputShape))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		# second set of CONV => RELU => POOL layers
		model.add(Conv2D(50, (5, 5), padding="same"))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		# first (and only) set of FC => RELU layers
		model.add(Flatten())
		model.add(Dense(500))
		model.add(Activation("relu"))

		# softmax classifier
		model.add(Dense(classes))
		model.add(Activation("softmax"))

		# return the constructed network architecture
		return model

# Alex net, from catdog and powerpoint #13
#def AlexNet(width=1, height=1, depth=1, classes=1):
#    model = Sequential()
#    inputshape = (height,width,depth)
#    # first layer convulional layer 
#    model.add(Con2D(filters=96,input_shape=inputshape,kernel_size=(3,3),stride=4,padding='valid'))
#    model.add(Activation('relu'))
#    # second layer, pooling 3X3 applyed with a stride of 2
#    model.add(MaxPolling2D(pool_size=(3,3),stride=2,padding='valid'))
#    # third layer, batch normalization
#    # https://towardsdatascience.com/batch-normalization-in-neural-networks-1ac91516821c
#    model.add(BatchNormalization())
#    # fourth layer, 256 5X5 filters with a stride of 1 and a padding of 2
#    model.add(Conv2D(filters=256,kernel_size=(5,5),stride=1,padding='valid'))
#    model.add(Activation('relu'))
#    return model
    
# note, this is my instance of the alenet
# the         
def AlexNet(width, height, depth, classes):
    model = Sequential()
    inputshape = (width,height,depth)
    inputvolume = width*height*depth
    # first layer 
    model.add(Conv2D(filters=96, input_shape=inputshape, kernel_size=(11,11), strides=(4,4), padding="valid"))
    model.add(Activation("relu"))
    model.add(BatchNormalization())

    # second layer, max pooling
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding="valid")) # in powerpoint pool size is 3,3
    # thierd layer, second convultion
    model.add(Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), padding="valid")) # kernal used to be 11x11
    model.add(Activation("relu"))
    model.add(BatchNormalization())

    # skipping normalization, for now
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding="valid")) # again pool size is 3,3 in power point
    # fourth layer, convultional layer 
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding="valid"))
    model.add(Activation("relu"))
    model.add(BatchNormalization())

    # fith layer convultional layer 
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding="valid"))
    model.add(Activation("relu"))
    model.add(BatchNormalization())

    
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="valid"))
    model.add(Activation("relu"))
    model.add(BatchNormalization())

    # max pooling...
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding="valid"))
    # at this point we need to start getting ready for the interconnected layer 
    model.add(Flatten()) # flatten from a 3d shape into a 2d shape
    # fully connected layer, poweroint has two of these
    model.add(Dense(4096, input_shape=(inputvolume,)))
    model.add(Activation("relu"))
    model.add(Dropout(0.4)) # to prevent overfitting
    
    model.add(Dense(4096,))
    model.add(Activation("relu"))
    model.add(Dropout(0.4)) # to prevent overfitting
    
    # 3rd Dense Layer
    model.add(Dense(1000))
    model.add(Activation("relu"))
    # Add Dropout
    model.add(Dropout(0.4))
    # Batch Normalisation
    #model.add(BatchNormalization())
    
    model.add(Dense(classes)) # should have 3 classes, for some reason only works with 2
    #model.add(Activation("relu"))
    model.add(Activation("softmax"))
    return model

# OLD stuff
# download all the contents of a folder into a numpy matrix
# define the paths ...
#imgpath = "./../Midterm_Project_Data" # reltive link to the image files 
#imgpath = "./Midterm_Project_Data"
#testdata_path = imgpath + "/test"
#trainingdata_path = imgpath + "/train"
#background = "/background"
#mal = "/mal"
#ben = "/ben"
#testben = testdata_path + ben
#testmal = testdata_path + mal
#testbackground = testdata_path + background
## paths for training data 
#trainben = trainingdata_path + mal
#trainmal = trainingdata_path + ben
#trainbackground = trainingdata_path + background
#
## load all the images from their respective file
#print("loading test and training images...")
#test_benign = loadfolder(testben)
#train_benign = loadfolder(trainben)
#test_malig = loadfolder(testmal)
#train_malig = loadfolder(trainmal)
#train_background = loadfolder(trainbackground)
#test_background = loadfolder(testbackground)

# new stuff
# figure out the folder names 
datafolder = "./Midterm_Project_Data"
train_dir = os.path.join(datafolder, "train")
test_dir = os.path.join(datafolder, "test")
# load the data using keras's build in stuff
batch_size = 64
datagen_train = ImageDataGenerator(rescale=1./255)
datagen_test = ImageDataGenerator(rescale=1./255)

print("loading images:")

generator_train = datagen_train.flow_from_directory(directory=train_dir,
                                                    batch_size=batch_size,
                                                    target_size=(256,256),
                                                    shuffle = True,
                                                    class_mode = 'categorical')

generator_test = datagen_test.flow_from_directory(  directory=test_dir,
                                                    batch_size=batch_size,
                                                    target_size=(256,256),
                                                    class_mode = 'categorical',
                                                    shuffle = False)



print("done loading images")
#print("now training network...")
testclasses = 3
lenetm1 = LeNet(width=256, height=256, depth=3, classes=testclasses) # define an instance of an LeNet, just a test
alexnm1 = AlexNet(width=256, height=256, depth=3, classes=testclasses) # define the model for Alex Net
##print("alexnet summery")
#lenetm1.summary() # prints out a summery of the network
##alexnm1.summary()
# compile the network
alexnm1.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
#lenetm1.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
## train the network
#epochs = 50
#batchsize = 25 
##Adam = keras.optimizers.adam(lr=0.001, amsgrad = True)
#steps_test = generator_test.n // batch_size
#steps_per_epoch = generator_train.n // batch_size
#
## do the actuall training...
#model = lenetm1
#history= model.fit_generator(generator_train,
#                           epochs=epochs,
#                           steps_per_epoch=steps_per_epoch,
#                           validation_data = generator_test,
#                           validation_steps = steps_test)

def trainNewModel():
    epochs = 25 # switched to 25 b/c 50 is too big
    #batchsize = 25 # not really needed in this function
    steps_test = generator_test.n // batch_size
    steps_per_epoch = generator_train.n // batch_size
    #model = lenetm1 # using lenet... for now 
    model = alexnm1 
    history= model.fit_generator(generator_train,
                           epochs=epochs,
                           steps_per_epoch=steps_per_epoch,
                           validation_data = generator_test,
                           validation_steps = steps_test)
    
    return (history,model)

# not really needed, but probubly gonna keep
def runClassifier():
    pass

# code the serilizing and loading history as a file
# this will pretty much work for any class type 
def saveHistory(history):
    bhis = pickle.dumps(history)
    hfile = open("history.bin","wb")
    hfile.write(bhis)
    hfile.close()
    
def loadHistory():
    bfile = open("history.bin","rb")
    bdata = bfile.read()
    bfile.close()
    return pickle.loads(bdata)

# takes in the history object generated by model fit as an argument
# runs the graphing function from the catdog inclass exsersise 
def showGraphs(history):
    plt.figure(figsize=[8,6])
    plt.plot(history.history['loss'],'r',linewidth=3.0)
    plt.plot(history.history['val_loss'],'b',linewidth=3.0)
    plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Loss',fontsize=16)
    plt.title('Loss Curves',fontsize=16)

    plt.figure(figsize=[8,6])
    plt.plot(history.history['acc'],'r',linewidth=3.0)
    plt.plot(history.history['val_acc'],'b',linewidth=3.0)
    plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Accuracy',fontsize=16)
    plt.title('Accuracy Curves',fontsize=16)

if __name__ == "__main__":
    history = None
    currentModel = None
    filename = "mt_weightmatrix.sav" # file name for the saved model
    print("mashine learning midterm")
    # simple cli control for controlling this thing
    ans = input("generate and classify new model? (y/n)) >")
    if ans == "y":
        print("training and classifying new model...")
        history,currentModel = trainNewModel()
        currentModel.save(filename) # I think this just assumes local file path
        saveHistory(history) # save the history of the training session to a file
        print("done training and saving model")
    else:
        print("loading files from memory, if they exisit")
        if os.path.isfile(filename) and os.path.isfile("history.bin"):# check to see if the file exists 
            currentModel = load_model(filename)
            history = loadHistory() # load history from an object
            print("done loading files")
        else:
            print("no files loaded")
    
    
    ans = input("display huristics? (y/n) >")
    if ans == "y":
        print("printing graphs")
        showGraphs(history)
        
    print("done!")