# -*- coding: utf-8 -*-
"""
Created on Fri Dec 04 13:04:52 2020

@author: kly
"""
import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras import Model

np.set_printoptions(threshold=np.inf)
class NN2(Model):
    def __init__(self):
        super(NN2,self).__init__()
        self.con1 = Conv2D(filters=32, kernel_size=(3, 3),
                         activation='relu')
        self.pol1 = MaxPool2D(pool_size=(2, 2))

        self.con2 = Conv2D(filters=64, kernel_size=(3, 3),
                         activation='relu')
        self.pol2 = MaxPool2D(pool_size=(2, 2))
        
        self.con3=Conv2D(filters=128, kernel_size=(3, 3),
                         activation='relu')
        self.pol3 = MaxPool2D(pool_size=(2, 2))

        self.flatten = Flatten()
        self.full1 = Dense(128, activation='relu')
        self.d1=Dropout(0.5)                       #dropout 
        self.full2 = Dense(64, activation='relu')
        self.d2=Dropout(0.5)
        self.full3 = Dense(1, activation='sigmoid')
    def call(self,x):
        x=self.con1(x)
        x=self.pol1(x)
        x=self.con2(x)
        x=self.pol2(x)
        x=self.con3(x)
        x=self.pol3(x)
        
        x=self.flatten(x)
        x=self.full1(x)
        x=self.d1(x)
        x=self.full2(x)
        x=self.d2(x)
        y=self.full3(x)
        return y