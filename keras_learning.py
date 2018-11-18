#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 22:13:31 2018

@author: dillon
"""

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten,Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

# setup test modles 
# truth table is a xor gate 
truthtable = np.array([[0,0],[0,1],[1,0],[1,1]],"float32") # inputs 
outputs = np.array([[0],[1],[1],[0]],np.float32) # outputs
lables = np.array([0,1]) # lables
# setup the Karas model
model = Sequential()
#model.add(Dense(16,input_dim=2)) # first layer 
#model.add(Activation('relu'))
model.add(Dense(16, input_dim=2, activation='relu'))
model.add(Dense(16, input_dim=2, activation='relu'))
model.add(Dense(16, input_dim=2, activation='relu'))
#model.add(Dense(1)) # output layer 
#model.add(Activation('sigmoid'))
model.add(Dense(1, activation='sigmoid'))
model.summary()
model.compile(optimizer='adam',loss='mean_squared_error',metrics=['binary_accuracy'])
#one_hot_labels = keras.utils.to_categorical(lables, num_classes=2)
#model.fit(outputs,truthtable,epochs=10)
model.fit(truthtable,outputs,epochs=10,verbose=2)

#for i in range(0,4):
#    ia = input("A: >")
#    ib = input("B: >")
#    nip = np.array([[ia],[ib]],"float32")
#    print(model.predict(nip.T).round())

ia = 1
ib = 1
inputdata = np.array([[ia],[ib]],"float32")
print("***********\ninput A: ",ia," input B: ",ib)
print("output: ",model.predict(inputdata.T).round())
model.summary()