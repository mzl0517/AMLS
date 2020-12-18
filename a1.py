# -*- coding: utf-8 -*-
"""
Created on Fri Dec 04 10:40:38 2020

@author:zhaolong
"""
import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras import Model

np.set_printoptions(threshold=np.inf)

class NN1(Model):
    def __init__(self):           #sturcture of convolutional nerual network
        super(NN1,self).__init__()
        self.con1 = Conv2D(filters=32, kernel_size=(3, 3), #first convolutinal bolck  kernel‘s size3*3
                         activation='relu')                # ReLu activation function
        self.pol1 = MaxPool2D(pool_size=(2, 2))            # first pooling layer 

        self.con2 = Conv2D(filters=32, kernel_size=(3, 3), # second convolutinal block
                         activation='relu')
        self.pol2 = MaxPool2D(pool_size=(2, 2))
        
        self.con3=Conv2D(filters=64, kernel_size=(3, 3),   # thrid convolutional block
                         activation='relu')
        self.pol3 = MaxPool2D(pool_size=(2, 2))

        self.flatten = Flatten()                      # one-dimentional
        self.full1 = Dense(128, activation='relu')    # fully-connected layers
        self.full2 = Dense(64, activation='relu')     
        self.full3 = Dense(1, activation='sigmoid')   #output block  
        
    
    def call(self,x):                                #call/generate the CNN
        x=self.con1(x)
        x=self.pol1(x)
        x=self.con2(x)
        x=self.pol2(x)
        x=self.con3(x)
        x=self.pol3(x)
        
        x=self.flatten(x)
        x=self.full1(x)
        x=self.full2(x)
        y=self.full3(x)
        return y
    """
    def predict(self,x):
        pass
    """




