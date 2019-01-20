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
count = 0
################################################################################
# Path to Dataset  
with open('train.csv', mode='r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    while count <= (9664*2):
        row = next(csv_reader)
#        for row in csv_reader:
        file = row[0]
        if row[1] == 'new_whale':
            label.append(row[1])
        else:
            label.append('Not_new_whale')
        im = cv2.imread(path1 + file)   
        img = cv2.resize(im, (img_rows, img_cols))
        gray = array(img).flatten()
        train.append(gray)
        print(i)
        i = i+1
        count = count + 1
      
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
nb_epoch = 100

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
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = 4)
X_train = x
X_train = X_train.reshape(x.shape[0], img_rows, img_cols, 3)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 3)

# Assigning X_train and X_test as float
X_train = X_train.astype('float32') 
X_test = X_test.astype('float32')
#######################################################################################
# Normalization of data 
# Data pixels are between 0 and 1
X_train /= 255
X_test /= 255

# Convert class vectors to binary class matrices
Y_train = y
Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

# Implementing a LeNet-5 model
model = Sequential()

model.add(Convolution2D(nb_filters_1, kernel_size = (nb_conv, nb_conv), activation='relu', input_shape=(img_rows, img_cols, 3)))
model.add(MaxPooling2D(pool_size = (nb_pool, nb_pool), strides = (2, 2)))

model.add(Convolution2D(nb_filters_2, kernel_size = (nb_conv, nb_conv), activation='relu', input_shape=(img_rows, img_cols, 3)))
model.add(MaxPooling2D(pool_size = (nb_pool, nb_pool), strides = (2, 2)))

model.add(Flatten())
model.add(Dense(500, activation='relu'))

model.add(Dense(nb_classes, activation='softmax'))

# Optimizer used is Stochastic Gradient Descent with learning rate of 0.01
# Loss is calculated using categorical cross entropy
opt = SGD(lr = 0.01)
model.compile(loss='binary_crossentropy', optimizer = opt, metrics = ['accuracy'])
callbacks = [EarlyStopping(monitor='val_loss', patience=5)]         
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
        im = cv2.imread(path + test_img)   
        img = cv2.resize(im, (img_rows,img_cols))
        gray = array(img).flatten()
        test.append(gray)
        print(j)
        j = j+1

#########################################################################
plt.figure(figsize=(8, 8))
plt.title("Learning curve")
plt.plot(hist.history["loss"], label="loss")
plt.plot(hist.history["val_loss"], label="val_loss")
plt.plot( np.argmin(hist.history["val_loss"]), np.min(hist.history["val_loss"]), marker="x", color="r", label="best model")
plt.xlabel("Epochs")
plt.ylabel("loss")
plt.legend();

#########################################################################
X_test = array(X_test)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 3)

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
score = model.evaluate(X_test, Y_test, verbose=0) # accuracy check
print('Test accuracy:', score[1])
res = []
################################################################################
for answers in range(0, len(p)):
    ans = copy(p[answers])
    for w in range(0,5):
        idx = argmax(ans)
        res.append(idx)
        ans[idx] = -1
    ans = 0
    
# res = reshape(res, [7960, 5])
        
# res = array(res)
csvfin = []
for r in range(0, len(res)):
    csvfin.append(list(encoding.keys())[list(encoding.values()).index(res[r])])
    
np.save('csvfin', array(csvfin))    
    
csvfin = reshape(csvfin, [7960, 5])
 
df = pd.DataFrame(data=csvfin[0:,0:])
       
df['Id'] = df[df.columns[0:]].apply(
   lambda x: ' '.join(x.dropna().astype(str)),
   axis=1)

df2 = df[['Id']]
#df2 = df2.as_matrix()

file2 = pd.DataFrame(data=(file_test[0:]))

myData = pd.concat([file2, df2], axis=1, sort=False)
myData.to_csv('kaggle_whale.csv')

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
