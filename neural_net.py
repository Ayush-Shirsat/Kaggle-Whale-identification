# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 15:54:58 2019

@author: Ayush Shirsat
"""

# Implementing a LeNet-5 Network to classify handwritten currency symbols

import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam
from keras.utils import np_utils
from keras.models import model_from_json
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Disable warning message of tensorflow
from PIL import Image
from numpy import *
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
import csv
import cv2
import pandas as pd

# paths
path1 = "./train/"
path2 = "./train_gray/"

# input image dimensions
img_rows, img_cols = 128, 128

# We have 1 channel as images are gray scale 
img_channels = 1
label = []
train = []
i = 1
################################################################################
# Path to Dataset  
with open('train.csv', mode='r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        file = row[0]
        label.append(row[1])
        im = Image.open(path1 + file)   
        img = im.resize((img_rows,img_cols))
        gray = array(img.convert('L')).flatten()
        train.append(gray)
        print(i)
        i = i+1
        
          
        # gray.save(path2 +  file, "JPEG")
        
        #print(row[0])
##########################################################################################
#imlist = os.listdir(path2)
#imlist = sorted(imlist)

#im1 = array(Image.open(path2 + imlist[0])) # Open one image to get size
#m,n = im1.shape[0:2] # Get the size of the images
#imnbr = len(imlist) # Get the number of images

# Create matrix to store all flattened images (i.e. image pixels are stored as a single row)
# Each row represents an image
#immatrix = array([array(Image.open(path2 + im2)).flatten() for im2 in imlist],'f')
####################################################################################
train = array(train)
label = array(label)

label = list(label)
label2 = pd.factorize(label)
label = label2[0]
l_name = label2[1]
encoding = {}
for i in range(0,len(l_name)):
    encoding.update({str(l_name[i]): i})

# Shuffling data and label together in a random order
data,Label = shuffle(train,label, random_state = 4)

#batch_size to train
batch_size = 256
# number of output classes
temp = set(Label)
nb_classes = len(temp)
print(nb_classes)
# number of epochs to train
nb_epoch = 5

# number of convolutional filters to use
nb_filters_1 = 20
nb_filters_2 = 50

# size of pooling area for max pooling
nb_pool = 2

# convolution kernel size
nb_conv = 5

# x is data & y is label
(x, y) = (data, Label)

# Split x and y into training and testing sets in random order
# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = 4)
X_train = x
X_train = X_train.reshape(x.shape[0], img_rows, img_cols, 1)
# X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

# Assigning X_train and X_test as float
X_train = X_train.astype('float32') 
# X_test = X_test.astype('float32')
#######################################################################################
# Normalization of data 
# Data pixels are between 0 and 1
X_train /= 255
# X_test /= 255

# Convert class vectors to binary class matrices
Y_train = y
Y_train = np_utils.to_categorical(Y_train, nb_classes)
# Y_test = np_utils.to_categorical(y_test, nb_classes)

# Implementing a LeNet-5 model
model = Sequential()

model.add(Convolution2D(nb_filters_1, kernel_size = (nb_conv, nb_conv), activation='relu', input_shape=(img_rows, img_cols, 1)))
model.add(MaxPooling2D(pool_size = (nb_pool, nb_pool), strides = (2, 2)))

model.add(Convolution2D(nb_filters_2, kernel_size = (nb_conv, nb_conv), activation='relu', input_shape=(img_rows, img_cols, 1)))
model.add(MaxPooling2D(pool_size = (nb_pool, nb_pool), strides = (2, 2)))

model.add(Flatten())
model.add(Dense(500, activation='relu'))

model.add(Dense(nb_classes, activation='softmax'))

# Optimizer used is Stochastic Gradient Descent with learning rate of 0.01
# Loss is calculated using categorical cross entropy
opt = SGD(lr = 0.01)
model.compile(loss='categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])
callbacks = [EarlyStopping(monitor='val_acc', patience=5)]         
# Starts training the model      
hist = model.fit(X_train, Y_train, batch_size = batch_size, epochs = nb_epoch, verbose = 1, validation_split = 0.1, shuffle = True)

# visualizing losses and accuracy
train_loss = hist.history['loss']
val_loss = hist.history['val_loss'] 
train_acc = (hist.history['acc'])
val_acc = (hist.history['val_acc'])
#########################################################################
path = "./test/"
test = []
file_test = os.listdir("test")
j = 1
for row in file_test:
        test_img = row
        im = Image.open(path + test_img)   
        img = im.resize((img_rows,img_cols))
        gray = array(img.convert('L')).flatten()
        test.append(gray)
        print(j)
        j = j+1


#########################################################################
X_test = array(test)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

# Assigning X_train and X_test as float
X_test = X_test.astype('float32') 

X_test /= 255
#######################################################################################
# Normalization of data 
# Data pixels are between 0 and 1


# score = model.evaluate(X_test, Y_test, verbose=0) # accuracy check
# print('Test accuracy:', score[1]) # Prints test accuracy
 
y_pred = model.predict_classes(X_test) # Predicts classes of all images in test data 

p = model.predict_proba(X_test) # To predict probability
res = []

for answers in range(0, len(p)):
    ans = p[answers]
    for w in range(0,5):
        idx = argmax(ans)
        res.append(idx)
        ans[idx] = 0
    ans = 0
        
        
    
#print('\nConfusion Matrix')
#print(confusion_matrix(np.argmax(Y_test,axis=1), y_pred)) # Prints Confusion matrix for analysis

# Serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# Serialize weights to H5
model.save_weights("model.h5")
print("Saved model to disk")

# X_test and Y_test are saved so model can be tested 
np.save('X_test', X_test)
np.save('Y_test', Y_test)
