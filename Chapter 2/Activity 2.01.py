# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 01:19:51 2020

@author: PRAJWAL
"""
#Simple ANN model using keras
import pandas as pd

dataset = pd.read_csv('OSI_dataset_e3.csv')
dataset.shape

feature = pd.read_csv('OSI_feature_e3.csv')
feature.shape

target = dataset['Revenue']

#splitting the data into training and testing
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(feature,target,test_size=0.2,random_state=42)

from keras.models import Sequential
from keras.layers import Dense,Activation

#Model creation
model = Sequential()

#Single layer ANN
model.add(Dense(1,input_dim=69))
model.add(Activation('sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

history = model.fit(x=X_train, y=y_train, epochs=10, batch_size=32, validation_split=0.2, shuffle=False)

import matplotlib.pyplot as plt
%matplotlib inline
plt.plot(history.history['loss'])
plt.show()

#predicting the test dataset
test_loss = model.evaluate(X_test,y_test)

