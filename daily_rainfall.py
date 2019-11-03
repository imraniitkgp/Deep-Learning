# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 19:16:56 2019

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

values = pd.read_excel('Orissa_cities1.xlsx', index_col='Date', header = 0)

dataset = array(values)
plt.figure(figsize=(50,10))
plt.matshow(values.corr(), fignum=1)
plt.xticks(range(len(values.columns)), values.columns)
plt.yticks(range(len(values.columns)), values.columns)
plt.colorbar()
plt.show()

min_max = np.empty([11,2])
dataset_scaled = np.empty([23725,11])
for i in range(11):
    min_max[i,0] = max(dataset[:,i])
    min_max[i,1] = min(dataset[:,i])
    for j in range(23725):
        dataset_scaled[j,i] = (dataset[j,i]-min_max[i,0])/(min_max[i,0]-min_max[i,1])
    

n=4720
lag=5
# splitting data into two parts
data1 = dataset_scaled[0:-n]
data2 = dataset_scaled[-n:]

# Data Preperation for Training
# Data Preperation for Training
train_x = []
train_y = []
for i in range(lag,23725-n):
    train_x.append(data1[i-lag:i, 0:11])
    train_y.append(data1[i, 0:1])
train_x = np.array(train_x)
train_y = np.array(train_y)
train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1],train_x.shape[2]))


# summarize the data
for i in range(25):
    print(train_x[i], train_y[i])


# Data Preperation for Testing
test_input = dataset_scaled[len(dataset_scaled)-len(data2)-lag:]
test_x = []
test_y=[]
for j in range(lag, len(data2)+lag):
    test_x.append(test_input[j-lag:j, 0:11])
    test_y.append(test_input[j, 0:1])
test_x = np.array(test_x)
test_y = np.array(test_y)
test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], train_x.shape[2]))


# Define Model
model = Sequential()   
model.add(LSTM(units = 120, return_sequences = True, input_shape = (train_x.shape[1], train_x.shape[2])))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(LSTM(units = 120, return_sequences = True))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(LSTM(units = 120, return_sequences = True))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(LSTM(units = 120, return_sequences = True))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(LSTM(units = 120, return_sequences = False))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add((Dense(units = 1)))
model.compile(optimizer = opt , loss = 'mse',  metrics = ['mse'])
history = model.fit(train_x, train_y, epochs = 600, batch_size = 100, validation_data = (test_x, test_y), verbose = 1)
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
Training_scaled_value = data1[5:19005, 0]
Testing_scaled_value = data2[:, 0]
Actual_Value_Training = (Training_scaled_value*(min_max[0,0]-min_max[0,1]))+min_max[0,0]
Actual_Value_Testing = (Testing_scaled_value*(min_max[0,0]-min_max[0,1]))+min_max[0,0]

Predicted_Value_Training = (predictions_train *(min_max[0,0]-min_max[0,1]))+min_max[0,0]
Predicted_Value_Testing = (predictions_test*(min_max[0,0]-min_max[0,1]))+min_max[0,0]

Training__result = np.empty([19000,2])
Training__result[:,0] = Actual_Value_Training
Training__result[:,1] = Predicted_Value_Training[:,0]

Testing__result = np.empty([4720,2])
Testing__result[:,0] = Actual_Value_Testing
Testing__result[:,1] = Predicted_Value_Testing[:,0]

#plotting the training result (Time Series)
plt.figure(figsize=(14,5))
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
plt.figure(figsize=(14,5))
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
plt.legend()
plt.show()

np.corrcoef(Testing__result[:, 0], Testing__result[:, 1])
np.corrcoef(Training__result[:, 0], Training__result[:, 1])