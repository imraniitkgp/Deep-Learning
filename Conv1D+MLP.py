# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 12:39:13 2019

@author: civil
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import adam
from numpy import array
import tensorflow as tf
import random as rn
import os
from keras.layers.convolutional import Conv1D
from keras.layers import Flatten
from keras.layers.convolutional import MaxPooling1D


os.environ['PYTHONHASHSEED'] = '0'

# Setting the seed for numpy-generated random numbers
np.random.seed(37)

# Setting the seed for python random numbers
rn.seed(1254)

# Setting the graph-level random seed.
tf.set_random_seed(89)

from keras import backend as K

session_conf = tf.ConfigProto(
      intra_op_parallelism_threads=1,
      inter_op_parallelism_threads=1)

#Force Tensorflow to use a single thread
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)

K.set_session(sess)

opt = adam(lr=0.00001) 

values = pd.read_excel('Orissanormalized_cities1.xlsx', index_col='Date', header = 0)

dataset = array(values)
plt.figure(figsize=(50,10))
plt.matshow(values.corr(), fignum=1)
plt.xticks(range(len(values.columns)), values.columns)
plt.yticks(range(len(values.columns)), values.columns)
plt.colorbar()
plt.show()

#min_max = np.empty([11,2])
#dataset_scaled = np.empty([23725,11])
#for i in range(11):
#    min_max[i,0] = max(dataset[:,i])
#    min_max[i,1] = min(dataset[:,i])
#    for j in range(23725):
#        dataset_scaled[j,i] = (dataset[j,i]-min_max[i,0])/(min_max[i,0]-min_max[i,1])
    

n=4720
# splitting data into two parts
data1 = dataset[0:-n]
data2 = dataset[-n:]

#number of input (days)
lag=5
#output (days)
n_output = 1

variable = data1.shape[1]
# Data Preperation for Training
train_x = np.empty([data1.shape[0]-(lag+n_output-1), lag*variable])
train_y = np.empty([data1.shape[0]-(lag+n_output-1), n_output])
for i in range(len(data1)-(lag+n_output-1)):
    for j in range(0,lag):
        m = variable*j
        train_x[i,m:m+variable] = data1[i+j,:]
    train_y[i,:] = data1[i+lag:i+lag+n_output,0] 
train_x = train_x.reshape(train_x.shape[0], 1, train_x.shape[1])

# visualisation of the prepared training dataset
for i in range(25):
    print(train_x[i], train_y[i])


# Data Preperation for Testing
test_input = dataset[len(dataset)-len(data2)-lag:]
test_x = np.empty([test_input.shape[0]-(lag+n_output-1), lag*variable])
test_y = np.empty([test_input.shape[0]-(lag+n_output-1), n_output])
for i in range(len(test_input)-(lag+n_output-1)):
    for j in range(0,lag):
        m = variable*j
        test_x[i,m:m+variable] = test_input[i+j,:]
    test_y[i,:] = test_input[i+lag:i+lag+n_output,0] 
test_x = test_x.reshape(test_x.shape[0], 1, test_x.shape[1])



# Define Model
model = Sequential()
model.add(Conv1D(filters=32,kernel_size=1, activation='relu',input_shape = (train_x.shape[1], train_x.shape[2])))   
#model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=32, kernel_size=1, activation='relu'))
model.add(Conv1D(filters=32, kernel_size=1, activation='relu'))
model.add(Conv1D(filters=32, kernel_size=1, activation='relu'))
#model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(units = 40, return_sequences = True ))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(LSTM(units = 40, return_sequences = True ))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(LSTM(units = 40, return_sequences = True))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(LSTM(units = 40, return_sequences = True))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
#model.add(Dense(100, activation='relu'))
model.add((Dense(units = 1)))
model.compile(optimizer = 'Adam' , loss = 'mse',  metrics = ['mse'])
history = model.fit(train_x, train_y, epochs = 100, batch_size = 300, validation_data = (test_x, test_y), verbose = 1)
model.summary()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model train vs validation loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(['train', 'validation'])
#plt.savefig('loss curve.png')
plt.show()

#Model Prediction
predictions_test = model.predict(test_x)
predictions_train = model.predict(train_x)

# Converting the scaled value to actual value
Actual_Value_Training = data1[5:19005, 0:1]
Actual_Value_Testing = data2[:, 0:1]


Predicted_Value_Training = predictions_train
Predicted_Value_Testing = predictions_test




#plotting the training result (Time Series)
plt.figure(figsize=(10,5))
plt.plot(Actual_Value_Training[:], color = 'red', label = 'Observed Rainfall')
plt.plot(Predicted_Value_Training[:], color = 'blue', label = 'Predicted rainfall')
plt.title('Prediction Performance during Training')
plt.xlabel('Time(Days)')
plt.ylabel('Rainfall in mm')
plt.legend()
plt.show()

#plotting the training result (Scatter Plot)
plt.figure(figsize=(5,5))
plt.scatter(Actual_Value_Training, Predicted_Value_Training)
#plt.plot(Predicted_Value_Training[1:50], color = 'blue', label = 'Predicted Temperature')
plt.title('Scatter Plot during Training')
plt.xlabel('Time(Days)')
plt.ylabel('Rainfall')
plt.legend()
plt.show()

#plotting the testing result (Time Series)
plt.figure(figsize=(10,5))
plt.plot(Actual_Value_Testing[:], color = 'red', label = 'Observed Rainfall')
plt.plot(Predicted_Value_Testing[:], color = 'blue', label = 'Predicted Rainfall')
plt.title('Prediction Performance during Testing')
plt.xlabel('Time(Days)')
plt.ylabel('Rainfall in mm')
plt.legend()
plt.show()

#plotting the testing result (Scatter Plot)
plt.figure(figsize=(5,5))
plt.scatter(Actual_Value_Testing, Predicted_Value_Testing)
#plt.scatter(Predicted_Value_Testing[1:50], color = 'blue', label = 'Predicted Temperature')
plt.title('Scatter Plot during Testing')
plt.xlabel('Time(Days)')
plt.ylabel('Rainfall in mm')
#plt.legend()
plt.show()

np.corrcoef(Actual_Value_Training[:,0], Predicted_Value_Training[:,0])
np.corrcoef(Actual_Value_Testing[:, 0], Predicted_Value_Testing[:, 0])