# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 15:55:21 2020

@author: PRAJWAL
"""

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense,Activation

#import required packages for plotting
import matplotlib.pyplot as plt
import matplotlib

import matplotlib.patches as mpatches

#import the function for plotting decision boundary
from utils import plot_decision_boundary
from mlxtend.plotting import plot_decision_regions

#Loading the dataset
X = pd.read_csv('outlier_feats.csv')
y = pd.read_csv('outlier_target.csv')


class_1 = plt.scatter(X.loc[y['Class']==0,'feature1'],X.loc[y['Class']==0,'feature2'],s=40,c='red',edgecolor='k')
class_2 = plt.scatter(X.loc[y['Class']==1,'feature1'],X.loc[y['Class']==1,'feature2'],s=40,c='blue',edgecolor='k')
plt.legend((class_1,class_2),('Fail','Pass'))
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

#------------------------------------------------------------------------------------------
#Model 1


model_1 = Sequential()
model_1.add(Dense(1,activation='sigmoid',input_dim=2))
model_1.compile(optimizer='sgd',loss='binary_crossentropy',metrics=['accuracy'])

#Fit model_1
model_1.fit(X,y,batch_size = 5,epochs=100,validation_split=0.2,shuffle=False)

test_loss_1 = model_1.evaluate(X,y) 


#------------------------------------------------------------------------------------------
#Model 2
model_2 = Sequential()
model_2.add(Dense(3,activation='relu',input_dim=2))
model_2.add(Dense(1,activation='sigmoid'))
model_2.compile(optimizer='sgd',loss='binary_crossentropy',metrics=['accuracy'])
 
#Fit model 2
model_2.fit(X, y, batch_size=5, epochs=200, verbose=1, validation_split=0.2, shuffle=False)

test_loss_2 = model_2.evaluate(X,y)

#------------------------------------------------------------------------------------------
#Model 3


model_3 = Sequential()
model_3.add(Dense(6,activation='relu',input_dim=2))
model_3.add(Dense(1,activation='sigmoid'))
model_3.compile(optimizer='sgd',loss='binary_crossentropy',metrics=['accuracy'])
 
#Fit model 2
model_3.fit(X, y, batch_size=5, epochs=400, verbose=1, validation_split=0.2, shuffle=False)

test_loss_3 = model_3.evaluate(X,y)

#------------------------------------------------------------------------------------------
#Model 4

model_4 = Sequential()
model_4.add(Dense(3,activation='tanh',input_dim=2))
model_4.add(Dense(1,activation='sigmoid'))
model_4.compile(optimizer='sgd',loss='binary_crossentropy',metrics=['accuracy'])
 
#Fit model 2
model_4.fit(X, y, batch_size=5, epochs=200, verbose=1, validation_split=0.2, shuffle=False)

test_loss_4 = model_4.evaluate(X,y)

#------------------------------------------------------------------------------------------
#Model 5

model_5 = Sequential()
model_5.add(Dense(6,activation='tanh',input_dim=2))
model_5.add(Dense(1,activation='sigmoid'))
model_5.compile(optimizer='sgd',loss='binary_crossentropy',metrics=['accuracy'])
 
#Fit model 2
model_5.fit(X, y, batch_size=5, epochs=400, verbose=1, validation_split=0.2, shuffle=False)

test_loss_5 = model_5.evaluate(X,y)




