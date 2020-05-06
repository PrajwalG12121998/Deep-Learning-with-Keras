# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 14:28:29 2020

@author: PRAJWAL
"""
import numpy as np
import pandas as pd

feature = pd.read_csv('tree_class_feats.csv')
target = pd.read_csv('tree_class_target.csv')

from keras.models import Sequential
from keras.layers import Dense,Activation

#Defining the model
model = Sequential()

#Adding first hidden layer
model.add(Dense(10,activation='tanh',input_dim=10))

#Adding another hidden layer
model.add(Dense(5,activation='tanh'))

#Output layer
model.add(Dense(1,activation='sigmoid'))

#Compiling the model
model.compile(optimizer='sgd',loss='binary_crossentropy',metrics=['accuracy'])
model.summary()

history = model.fit(feature,target,epochs=100,batch_size=5,validation_split=0.2,shuffle=False,verbose=1)

#plotting the graph of accuracy and loss
import matplotlib.pyplot as plt
%matplotlib inline
#Plotting training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train','Validation'],loc='upper left')
plt.show()


#Plotting training &validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train','Validation'],loc='upper left')
plt.show()

#Predicting the first 10 input data
y_predicted = model.predict(feature.iloc[0:10,:])

test_loss = model.evaluate(feature.iloc[0:10,:],target.iloc[0:10,:])





