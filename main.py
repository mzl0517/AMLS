# -*- coding: utf-8 -*-
"""
Created on  Wen Dec 02 17:36:39 2020
main function
@author: Zhaolong
"""
import cv2 #OpenCV package, which is dedicated in image processing
import os  #used to get the path of current file
import numpy as np # matrix management ,calculation and manipulate data
import tensorflow as tf # machine learning platform
import os
import numpy as np
from matplotlib import pyplot as plt #matplotlib pyplot is used for visulatization
#functions used in keras
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense 
#Two dimensional convolutional neural network, for optimization results,activation function
from tensorflow.keras import Model
from A1.a1 import NN1
from A2.a2 import NN2
from B1.b1 import NN3
from B2.b2 import NN4
np.set_printoptions(threshold=np.inf)
####Gets the address and get all the images in the list file
def getFlist(path):
    flist= [] 
    lsdir = os.listdir(path)
    dirs = [i for i in lsdir if os.path.isdir(os.path.join(path, i))] 
    if dirs:
        for i in dirs:
            getFlist(os.path.join(path, i))
    files = [i for i in lsdir if os.path.isfile(os.path.join(path, i))]
    for file in files:
        flist.append(file)
    return flist
filename2=r"/Users/zhaolong/Desktop/AMLS_20-21_SN17065872(1)/Datasets/cartoon_set/img"
filename1=r"/Users/zhaolong/Desktop/AMLS_20-21_SN17065872(1)/Datasets/celeba/img"
flist1=getFlist(filename1)
flist2=getFlist(filename2)
#used to test if the program run and image have been readed
img1=cv2.imread(filename2+'/'+'1838.png') 
cv2.imshow("cat",img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
##### pre-processing 
data1=np.zeros((5000,64,64,3))  # size of feature , input data size
data2=np.zeros((10000,128,128,3)) #size of feature
for i in range(len(flist1)):
    img=cv2.imread(filename1+'/'+flist1[i])
    img=cv2.resize(img,(64,64),interpolation=cv2.INTER_AREA) #resize the image for neural network 查resize函数interpolation 
    data1[i,:,:,:]=img/255.

for i in range(len(flist2)):
    img=cv2.imread(filename2+'/'+flist2[i])
    img=cv2.resize(img,(128,128),interpolation=cv2.INTER_AREA)
    data2[i,:,:,:]=img/255.
#### get/read the label
labels11=[]       #load from filename3
labels12=[]       #load from filename3
labels21=[]       #load from filename4
labels22=[]       #load from filename4
filename3=r"/Users/zhaolong/Desktop/AMLS_20-21_SN17065872(1)/Datasets/celeba/img/labels.csv"
filename4=r"/Users/zhaolong/Desktop/AMLS_20-21_SN17065872(1)/Datasets/cartoon_set/img/labels.csv"
import csv
with open(filename3,'r') as f:
    reader = csv.reader(f)
    i=0
    for row in reader:
        if i!=0:
            tmp=row[0].split('\t')
            if tmp[2]=='1':
                labels11.append(1.)
            else:
                labels11.append(0.)
            if tmp[3]=='1':
                labels12.append(1.)
            else:
                labels12.append(0.)
        i=i+1
labels11=np.array(labels11).reshape((len(labels11),1))
labels12=np.array(labels12).reshape((len(labels12),1))
with open(filename4,'r') as f1:
    reader = csv.reader(f1)
    i=0
    for row in reader:
        if i!=0:
            tmp=row[0].split('\t')
            labels21.append(float(tmp[1]))
            labels22.append(float(tmp[2]))
        i=i+1
labels21=np.array(labels21).reshape((len(labels21),1))
labels22=np.array(labels22).reshape((len(labels22),1))           
#### divide 3:1:1 on training set,validation set and test set
from sklearn.model_selection import train_test_split   #(train_data,train_target,test_size,random_state (seeds))
xa1,x_test_a1,ya1,y_test_a1=train_test_split(data1,labels11,test_size=0.2,random_state=666)      #4:1
x_train_a1,x_val_a1,y_train_a1,y_val_a1=train_test_split(xa1,ya1,test_size=0.25,random_state=666)#3:1

x_train_a1=tf.convert_to_tensor(x_train_a1)    #‘convert_to_tensor’ is used to transfer Python ob-ject to ‘Tensor’ objects
y_train_a1=tf.convert_to_tensor(y_train_a1)
x_val_a1=tf.convert_to_tensor(x_val_a1)
y_val_a1=tf.convert_to_tensor(y_val_a1)

xa2,x_test_a2,ya2,y_test_a2=train_test_split(data1,labels12,test_size=0.2,random_state=666)
x_train_a2,x_val_a2,y_train_a2,y_val_a2=train_test_split(xa2,ya2,test_size=0.25,random_state=666)

xb1,x_test_b1,yb1,y_test_b1=train_test_split(data2,labels21,test_size=0.2,random_state=666)
x_train_b1,x_val_b1,y_train_b1,y_val_b1=train_test_split(xb1,yb1,test_size=0.25,random_state=666)

xb2,x_test_b2,yb2,y_test_b2=train_test_split(data2,labels22,test_size=0.2,random_state=666)
x_train_b2,x_val_b2,y_train_b2,y_val_b2=train_test_split(xb2,yb2,test_size=0.25,random_state=666)
###start training A1
model1 = NN1()
sgd=tf.keras.optimizers.SGD(lr=0.5)         #set the learning rate of optimizer
adam=tf.keras.optimizers.Adagrad(lr=0.01)   # adagrad optimizer
rms=tf.keras.optimizers.RMSprop(rho=0.5)    
model1.compile(optimizer="adam",            
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), #loss function 二进制分别的
              metrics=['accuracy'])

checkpoint_save_path = "./checkpoint/nn1.ckpt"       #bottle neck feature : save the optimized/advanced CNN stucture before the fully-connected layer/output block
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model1.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,   # function to load the bottle neck feature
                                                 save_weights_only=True,
                                                 save_best_only=True)

history1 = model1.fit(x_train_a1,
                      y_train_a1, 
                      batch_size=32,      #set the batch_size image number per epochs to 32
                      epochs=100, 
                      validation_data=(x_val_a1,y_val_a1),
                      validation_freq=1,  
                    callbacks=[cp_callback])
model1.summary()         # show the results(accuracy)

# Show acc and Loss curves for training and validation sets
acc = history1.history['accuracy']       
val_acc = history1.history['val_accuracy']
loss = history1.history['loss']                
val_loss = history1.history['val_loss']
plt.figure(figsize=(16,9))                  #generate the graph
plt.subplot(1, 2, 1)                        # the first graph in Row 1, column 2
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()                                # interruption
plt.subplot(1, 2, 2)                        # the second graph in Row 1, column 2
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
########start training a2
model2 = NN2()
model2.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),    
              metrics=['accuracy'])

checkpoint_save_path = "./checkpoint/nn2.ckpt"
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model2.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)

history2 = model2.fit(x_train_a2,
                      y_train_a2, 
                      batch_size=32,
                      epochs=100, 
                      validation_data=(x_val_a2,y_val_a2),
                      validation_freq=1,
                    callbacks=[cp_callback])
model2.summary()
acc = history2.history['accuracy']
val_acc = history2.history['val_accuracy']
loss = history2.history['loss']
val_loss = history2.history['val_loss']
plt.figure(figsize=(16,9))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
###### start training b1
model3=NN3()
model3.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),  
              metrics=['sparse_categorical_accuracy'])

checkpoint_save_path = "./checkpoint/nn3.ckpt"
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model3.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)

history3 = model3.fit(x_test_b1,
                      y_test_b1,
                      batch_size=32,
                      epochs=100, 
                      validation_data=(x_val_b1,y_val_b1), 
                      validation_freq=1,
                    callbacks=[cp_callback])
model3.summary()
acc = history3.history['sparse_categorical_accuracy']
val_acc = history3.history['val_sparse_categorical_accuracy']
loss = history3.history['loss']
val_loss = history3.history['val_loss']
plt.figure(figsize=(16,9))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
#######start training b2
model4=NN4([2,2,2,2])
model4.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

checkpoint_save_path = "./checkpoint/nn3.ckpt"
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model4.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)

history4 = model4.fit(x_train_b2,
                      y_train_b2, 
                      batch_size=32,
                      epochs=100, 
                      validation_data=(x_val_b2,y_val_b2),
                      validation_freq=1,
                    callbacks=[cp_callback])
model4.summary()
acc = history4.history['sparse_categorical_accuracy']
val_acc = history4.history['val_sparse_categorical_accuracy']
loss = history4.history['loss']
val_loss = history4.history['val_loss']
plt.figure(figsize=(16,9))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()




 