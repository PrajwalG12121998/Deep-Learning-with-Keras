# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 16:51:45 2020

@author: PRAJWAL
"""
import pandas as pd
colnames = ['CIC0', 'SM1_Dz(Z)', 'GATS1i', 'NdsCH', 'NdssC','MLOGP', 'LC50']
dataset = pd.read_csv('qsar_fish_toxicity.csv', sep=';', names=colnames)

X = dataset.drop('LC50',axis=1)
y = dataset['LC50']

from keras.models import Sequential
from keras.layers import Dense,Activation

def build_model():
    #build the model
    model = Sequential()
    model.add(Dense(8,activation='relu',input_dim=6))
    model.add(Dense(1))
    #compile the model
    model.compile(optimizer='adam',loss='mean_squared_error')
    #return the model
    return model

#scikit-Learn interface for the keras model
from keras.wrappers.scikit_learn import KerasRegressor

YourModel = KerasRegressor(build_fn=build_model,epochs=100,batch_size=20,verbose=1)

from sklearn.model_selection import cross_val_score
scores = cross_val_score(YourModel,X,y,cv=5)

print(abs(scores.mean()))

#LeaveOneOut cross-validation
from sklearn.model_selection import LeaveOneOut

loo = LeaveOneOut()
scores_loo = cross_val_score(YourModel,X,y,cv=loo)

print(scores_loo.mean())