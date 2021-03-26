# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 18:33:01 2021

@author: Nicka
"""

import matplotlib.image as mpimg # To read in images 
import matplotlib.pyplot as plot  # to print images 
import os # for file scraping 
import numpy as np # for matrix operations 
from PIL import Image # to handle images 
import pandas as pd # data processing 
import tensorflow as tf # for CNN 
from tensorflow.keras import layers, models # For CNN 
from keras import optimizers, losses, activations, models # For CNN 
from keras.layers import Convolution2D, Dense, Input, Flatten, Dropout, MaxPooling2D, BatchNormalization, GlobalMaxPool2D, Concatenate
from keras.models import Model
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Get the names and locations of all the files, 
start_folder = "C:/Users/Nicka/Documents/GradSchool/ADV_machineLearn/datasets/sp-society-camera-model-identification"

def get_paths(start_folder):
  list_paths = []
  for subdir, dirs, files in os.walk(start_folder):
      for file in files:
          filepath = subdir + os.sep + file
          list_paths.append(filepath)
          
  return list_paths
  
# get the training samples 
result_list=[]
result_list = get_paths(start_folder)
# sample 
print(result_list[500])

# Split the locations to testing / training 
list_train = [filepath for filepath in result_list if "train" in filepath]
list_test = [filepath for filepath in result_list if "test" in filepath]

# Print out a location from both the test and the train set 
print(list_train[1])
print("\n")
  
print(list_test[1])
print("\n")

# Get the name of the cameras 
camera_list =["HTC-1-M7","iPhone-4s","iPhone-6","LG-Nexus-5x","Motorola-Droid-Maxx","Motorola-Nexus-6","Motorola-X","Samsung-Galaxy-Note3","Samsung-Galaxy-S4","Sony-NEX-7"]

print(camera_list)

# get labels for training data  
label = [os.path.dirname(filepath).split(os.sep)[-1] for filepath in list_train]

# get x training values, Do not run if you do not have too!
# def read_image(filepath):
#   img_array = np.array(Image.open((filepath)), dtype="uint8") # unit8 contains all whole number between 0 to 255
#   pil_img = Image.fromarray(img_array) # creates image memory
#   return np.array(pil_img.resize((256,256)))/255 # resizes images and returns to x array
# 
# x_train = np.array([read_image(filepath) for filepath in list_train])
# np.save("x_train.npy", x_train)

# Get the testing values 
# x_test =np.array([read_image(filepath) for filepath in list_test])
# np.save("x_test.npy", x_test) # This needs to be tested

# if the images need to be reloaded... run this code! 
x_train = np.load("x_train.npy")
x_test = np.load("x_test.npy")

# Confirm both training and testing data was collected 
print(x_train[500])
print(x_test[500])
print(len(x_train))
print(len(x_test))

# Add in labels 
y_train = pd.get_dummies(pd.Series(label)) # converts one dimensional array into dummy variables 
label_ind = y_train.columns.values # each column represents a camera
y_train = np.array(y_train) # convert to numpy array 

# quick validation 
print(len(y_train[0]))
print((y_train))

# Use scikit learn to split training data into training and testing data.  
x_train_sub, x_test_sub, y_train_sub, y_test_sub = train_test_split(x_train,y_train, test_size=.15)

# Build CNN Model 
def get_model():
  start_shape = (256, 256, 3)
  num_class = len(label_ind)
  input_shape = Input(shape=start_shape)
  norm_inp = BatchNormalization()(input_shape)
  layers = Convolution2D(16, kernel_size=3, activation=activations.relu, padding="same")(norm_inp)
  layers = Convolution2D(16, kernel_size=3, activation=activations.relu, padding="same")(layers)
  layers = MaxPooling2D(pool_size=(3, 3))(layers)
  layers = Dropout(rate=0.2)(layers)
  layers = Convolution2D(32, kernel_size=3, activation=activations.relu, padding="same")(layers)
  layers = Convolution2D(32, kernel_size=3, activation=activations.relu, padding="same")(layers)
  layers = MaxPooling2D(pool_size=(3, 3))(layers)
  layers = Dropout(rate=0.2)(layers)
  layers = Convolution2D(64, kernel_size=2, activation=activations.relu, padding="same")(layers)
  layers = Convolution2D(20, kernel_size=2, activation=activations.relu, padding="same")(layers)
  layers = GlobalMaxPool2D()(layers) # Pools from whole input 
  layers = Dropout(rate=0.2)(layers)
  layers = Flatten()(layers)
  dense_1 = Dense(20, activation=activations.relu)(layers)
  dense_1 = Dense(num_class, activation=activations.softmax, name='get_dense')(dense_1)
  
  model = models.Model(inputs=input_shape, outputs=dense_1)
  opt = optimizers.Adam()
  
  model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])
  model.summary()
  return model
  
model = get_model()

# Get the fit history 
# This was ran twice
history = model.fit(x_train_sub, y_train_sub, epochs=37, validation_data = (x_test_sub, y_test_sub))

train_loss, train_acc = model.evaluate(x_train_sub,  y_train_sub, verbose=2)
test_loss, test_acc = model.evaluate(x_test_sub,  y_test_sub, verbose=2)
y_vals_cnn = model.predict(x_test_sub, verbose=2)
y_test_sub_argmax = np.argmax(y_test_sub,axis=1)
y_vals_cnn_argmax =  np.argmax(y_vals_cnn,axis=1)
print("Convolution Neural Network")
print(confusion_matrix(y_test_sub_argmax, y_vals_cnn_argmax))
print(classification_report(y_test_sub_argmax, y_vals_cnn_argmax))

# Get the model at final density layer. 
layer_name='get_dense'
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)

intermediate_layer_model.summary()

# New labels for xgboost
new_train = [None] * y_train_sub.shape[0]
i = 0
for i in range(0,y_train_sub.shape[0]):
  new_train[i] = y_train_sub[i,:].argmax()
  
new_val = [None] * y_test_sub.shape[0]
i = 0
for i in range(0,y_test_sub.shape[0]):
  new_val[i] = y_test_sub[i,:].argmax()
  
new_train = np.array(new_train)
new_val = np.array(new_val)

# get middle layer
# Use the CNN model that was already created. 
intermediate_output = intermediate_layer_model.predict(x_train_sub) 
intermediate_output = pd.DataFrame(data=intermediate_output)

val_data = intermediate_output[2300:]
val_data_label = new_train[2300:]

intermediate_test_output = intermediate_layer_model.predict(x_test_sub)
intermediate_test_output = pd.DataFrame(data=intermediate_test_output)

#Make XGBoost model
xgbmodel = XGBClassifier(objective='multi:softprob',
                      num_class= 10)
xgbmodel.fit(intermediate_output, new_train)
score = xgbmodel.score(intermediate_test_output, new_val)
y_pred_xgb = xgbmodel.predict(intermediate_test_output)

print("Convolution XGBoost")
print(confusion_matrix(new_val, y_pred_xgb))
print(classification_report(new_val, y_pred_xgb))
print(score)

#  Random forest 
rfModel = RandomForestClassifier()
rfModel.fit(intermediate_output, new_train)
y_pred_rf = rfModel.predict(intermediate_test_output)

score = accuracy_score(y_pred_rf, new_val)

print("Convolution Random Forest")
print(confusion_matrix(new_val,y_pred_rf))
print(classification_report(new_val, y_pred_rf))
print(score)

# Support Vector Machines 
svclassifier = SVC(kernel='poly')
svclassifier.fit(intermediate_output, new_train)

y_pred_svm = svclassifier.predict(intermediate_test_output)
score = accuracy_score(y_pred_svm, new_val)

print("Convolution Support Vector Machines")
print(confusion_matrix(new_val, y_pred_svm))
print(classification_report(new_val, y_pred_svm))
print(score)