# -*- coding: utf-8 -*-
"""
Created on Fri May  7 13:55:45 2021

@author: Ishwar
"""

import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Activation, Flatten
from keras.utils import np_utils
import matplotlib.pyplot as plt
import cv2
import numpy as np


X_train=np.empty((0,64,64),dtype='float32')
Y_train=np.zeros((0,2),dtype='float32')
import os

#Later check if 'X_train.append' is inefficient. If so, then use 'count'
for file in os.listdir("D:/ToyDataset/toy_train/circle"):
    if file.endswith(".jpg"):
        path=(os.path.join("D:/ToyDataset/toy_train/circle/", file))
        img=cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype('float32')       
        img=img.reshape(1,64,64)        
        X_train=np.append(X_train,img,axis=0)
        y_t=np.array([[1,0]])
        Y_train=np.append(Y_train,y_t,axis=0)
        
for file in os.listdir("D:/ToyDataset/toy_train/rectangle/"):
    if file.endswith(".jpg"):
        path=(os.path.join("D:/ToyDataset/toy_train/rectangle/", file))
        img=cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype('float32')       
        img=img.reshape(1,64,64)
        X_train=np.append(X_train,img,axis=0)
        y_t=np.array([[0,1]])
        Y_train=np.append(Y_train,y_t,axis=0)
        
X_train=X_train.reshape(X_train.shape[0],64,64,1)
#Y_train=Y_train.reshape(Y_train.shape[0],2,1)


#Build model
model = Sequential()

model.add(Conv2D(32, (3,3), input_shape=(64,64,1)))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(64,(3, 3)))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())
# Fully connected layer

model.add(Dense(512))
model.add(Activation('relu'))

model.add(Dropout(0.2))
model.add(Dense(2))
model.add(Activation('softmax'))

# looking at the model summary
model.summary()
# compiling the sequential model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
# training the model for 10 epochs
model.fit(X_train, Y_train, batch_size=50, epochs=10, validation_data=(X_train, Y_train))


#Sample testing
sample_x = X_train[999]

sample_show=sample_x.reshape(64,64)
plt.imshow(sample_show)
sample_x=sample_x.reshape(1,64,64,1)


out = model.predict(
    sample_x,
    batch_size=None,
    verbose='auto',
    steps=None,
    callbacks=None,
    max_queue_size=10,
    workers=1,
    use_multiprocessing=False
)
np.set_printoptions(precision=2)
print("Output is : ",out)

