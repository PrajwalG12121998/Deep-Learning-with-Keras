# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 13:15:05 2020

@author: PRAJWAL
"""

import pandas as pd
import numpy as np
from tensorflow import random
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

X = pd.read_csv('traffic_volume_feats.csv')
y = pd.read_csv('traffic_volume_target.csv')

#Output range
print("Output range:",y['Volume'].min()," - ",y['Volume'].max())

from keras.models import Sequential
from keras.layers import Dense,Activation

def build_model_1(optimizer='adam'):
    #create model 1
    model = Sequential()
    model.add(Dense(10,activation='relu',input_dim=X.shape[1]))
    model.add(Dense(1))
    #Compile model
    model.compile(loss='mean_squared_error',optimizer=optimizer)
    return model

def build_model_2(optimizer='adam'):
    #create model 1
    model = Sequential()
    model.add(Dense(10,activation='relu',input_dim=X.shape[1]))
    model.add(Dense(10,activation='relu'))
    model.add(Dense(1))
    #Compile model
    model.compile(loss='mean_squared_error',optimizer=optimizer)
    return model

def build_model_3(optimizer='adam'):
    #create model 1
    model = Sequential()
    model.add(Dense(10,activation='relu',input_dim=X.shape[1]))
    model.add(Dense(10,activation='relu'))
    model.add(Dense(10,activation='relu'))
    model.add(Dense(1))
    #Compile model
    model.compile(loss='mean_squared_error',optimizer=optimizer)
    return model

seed = 1
np.random.seed(seed)

from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import KFold,cross_val_score

results_1 = []
models = [build_model_1,build_model_2,build_model_3]

for i in range(len(models)):
    regressor = KerasRegressor(build_fn=models[i],verbose=1,batch_size=50,epochs=100,shuffle=False)
    model = make_pipeline(StandardScaler(),regressor)
    kf = KFold(n_splits=5,shuffle=True,random_state=seed)
    result = cross_val_score(model,X,y,cv=kf)
    results_1.append(result)

print("Cross-Validation Loss for Model 1 =", abs(results_1[0].mean()))
print("Cross-Validation Loss for Model 2 =", abs(results_1[1].mean()))
print("Cross-Validation Loss for Model 3 =", abs(results_1[2].mean()))

#Model 2 has lowest error

#-----------------------------------------------------------------------------------------------------------

np.random.seed(seed)

results_2 = []
epochs = [80,100]
batches = [50,25]

#Loop over epochs and batchsize
for e in range(len(epochs)):
    for b in range(len(batches)):
        regressor = KerasRegressor(build_fn=build_model_2,verbose=1,batch_size=batches[b],epochs=epochs[e],shuffle=False)
        model = make_pipeline(StandardScaler(),regressor)
        kf = KFold(n_splits=5)
        result = cross_val_score(model,X,y,cv=kf)
        results_2.append(result)

c = 0
for e in range(len(epochs)):
    for b in range(len(batches)):
        print("batch_size =", batches[b],", epochs =", epochs[e], ", Test Loss =", abs(results_2[c].mean()))
        c += 1

#-----------------------------------------------------------------------------------------------------------

np.random.seed(seed)
optimizers = ['adam','sgd','rmsprop']
results_3 = []

for o in range(len(optimizers)):
    optimizer = optimizers[o]
    regressor = KerasRegressor(build_fn=build_model_1,verbose=1,batch_size=50,epochs=100,shuffle=False)
    model = make_pipeline(StandardScaler(),regressor)
    kf = KFold(n_splits=5)
    result = cross_val_score(model,X,y,cv=kf)
    results_3.append(result)

# Print cross-validation score for each possible pair of optimizer, activation
c = 0
for o in range(len(optimizers)):
        print("optimizer = ", optimizers[o], ", Test Loss = ", abs(results_3[c].mean()))
        c += 1