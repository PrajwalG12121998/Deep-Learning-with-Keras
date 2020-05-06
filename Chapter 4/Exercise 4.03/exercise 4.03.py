# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 17:47:41 2020

@author: PRAJWAL
"""

# import data
import pandas as pd
import numpy as np
from tensorflow import random
colnames = ['CIC0', 'SM1_Dz(Z)', 'GATS1i', 'NdsCH', 'NdssC','MLOGP', 'LC50']
data = pd.read_csv('qsar_fish_toxicity.csv', sep=';', names=colnames)
X = data.drop('LC50', axis=1)
y = data['LC50']

#Defining 3 keras models
from keras.models import Sequential
from keras.layers import Dense,Activation

# 1 Hidden layer with size 4
def build_model_1(activation='relu',optimizer='adam'):
    # build the Keras model_1
    model = Sequential()
    model.add(Dense(4, input_dim=X.shape[1], activation='relu'))
    model.add(Dense(1))
    # Compile the model
    model.compile(loss='mean_squared_error', optimizer=optimizer,metrics=['accuracy'])
    # return the model
    return model

#1 Hidden layer with size 8
def build_model_2():
    # build the Keras model_2
    model = Sequential()
    model.add(Dense(8, input_dim=X.shape[1], activation='relu'))
    model.add(Dense(1))
    # Compile the model
    model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])
    # return the model
    return model

# 2 Hidden layer with size 4 and size 2
def build_model_3():
    # build the Keras model_3
    model = Sequential()
    model.add(Dense(4, input_dim=X.shape[1], activation='relu'))
    model.add(Dense(2, activation='relu'))
    model.add(Dense(1))
    # Compile the model
    model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])
    # return the model
    return model


seed = 1
np.random.seed(seed)

from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import KFold,cross_val_score


results_1 = []
models = [build_model_1,build_model_2,build_model_3]

for i in range(len(models)):
    model = KerasRegressor(build_fn=models[i],verbose=1,batch_size=20,epochs=100,shuffle=False)
    kf = KFold(n_splits=3)
    result = cross_val_score(model,X,y,cv=kf)
    results_1.append(result)

# print the cross-validation scores
print("Cross-Validation Loss for Model 1 =", abs(results_1[0].mean()))
print("Cross-Validation Loss for Model 2 =", abs(results_1[1].mean()))
print("Cross-Validation Loss for Model 3 =", abs(results_1[2].mean()))

#Model 1 gives the lowest error rate

np.random.seed(seed)

results_2 = []
epochs = [100,150]
batches = [20,15]

#Loop over epochs and batchsize
for e in range(len(epochs)):
    for b in range(len(batches)):
        model = KerasRegressor(build_fn=build_model_1,verbose=1,batch_size=batches[b],epochs=epochs[e],shuffle=False)
        kf = KFold(n_splits=3)
        result = cross_val_score(model,X,y,cv=kf)
        results_2.append(result)


# Print cross-validation score for each possible pair of epochs, batch_size
c = 0
for e in range(len(epochs)):
    for b in range(len(batches)):
        print("batch_size =", batches[b],", epochs =", epochs[e], ", Test Loss =", abs(results_2[c].mean()))
        c += 1

np.random.seed(seed)
activations = ['relu','tanh']
optimizers = ['adam','sgd','rmsprop']
results_3 = []

for o in range(len(optimizers)):
    for a in range(len(activations)):
        optimizer = optimizers[o]
        activation = activations[a]
        model = KerasRegressor(build_fn=build_model_1,verbose=1,batch_size=20,epochs=100,shuffle=False)
        kf = KFold(n_splits=3)
        result = cross_val_score(model,X,y,cv=kf)
        results_3.append(result)

# Print cross-validation score for each possible pair of optimizer, activation
c = 0
for o in range(len(optimizers)):
    for a in range(len(activations)):
        print("activation = ", activations[a],", optimizer = ", optimizers[o], ", Test Loss = ", abs(results_3[c].mean()))
        c += 1