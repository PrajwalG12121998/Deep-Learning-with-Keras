# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 11:29:08 2020

@author: PRAJWAL
"""

import pandas as pd
import numpy as np

X = pd.read_csv('HCV_feats.csv')
y = pd.read_csv('HCV_target.csv')

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

#Defining 3 keras models
from keras.models import Sequential
from keras.layers import Dense,Activation


def build_model_1():
    # build the Keras model_1
    model = Sequential()
    model.add(Dense(4, input_dim=X.shape[1], activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # return the model
    return model

def build_model_2(activation='relu',optimizer='adam'):
    # build the Keras model_1
    model = Sequential()
    model.add(Dense(4, input_dim=X.shape[1], activation=activation))
    model.add(Dense(2, activation=activation))
    model.add(Dense(1, activation='sigmoid'))
    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    # return the model
    return model
    
def build_model_3():
    # build the Keras model_1
    model = Sequential()
    model.add(Dense(8, input_dim=X.shape[1], activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # return the model
    return model

from tensorflow import random

seed = 1
np.random.seed(seed)

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold,cross_val_score


results_1 = []
models = [build_model_1,build_model_2,build_model_3]

for i in range(len(models)):
    model = KerasClassifier(build_fn=models[i],verbose=1,batch_size=20,epochs=100,shuffle=False)
    kf = KFold(n_splits=5)
    result = cross_val_score(model,X,y,cv=kf)
    results_1.append(result)

# print the cross-validation scores
print("Cross-Validation Accuracy for Model 1 =", abs(results_1[0].mean()))
print("Cross-Validation Accuracy for Model 2 =", abs(results_1[1].mean()))
print("Cross-Validation Accuracy for Model 3 =", abs(results_1[2].mean()))
#Model 2 gives the best accuracy 

#------------------------------------------------------------------------------------------------------------------------

np.random.seed(seed)

results_2 = []
epochs = [100,200]
batches = [10,20]

#Loop over epochs and batchsize
for e in range(len(epochs)):
    for b in range(len(batches)):
        model = KerasClassifier(build_fn=build_model_2,verbose=1,batch_size=batches[b],epochs=epochs[e],shuffle=False)
        kf = KFold(n_splits=5)
        result = cross_val_score(model,X,y,cv=kf)
        results_2.append(result)


# Print cross-validation score for each possible pair of epochs, batch_size
c = 0
for e in range(len(epochs)):
    for b in range(len(batches)):
        print("batch_size =", batches[b],", epochs =", epochs[e], ", Test Accuracy =", abs(results_2[c].mean()))
        c += 1

#------------------------------------------------------------------------------------------------------------------------
np.random.seed(seed)
activations = ['relu','tanh']
optimizers = ['adam','sgd','rmsprop']
results_3 = []

for o in range(len(optimizers)):
    for a in range(len(activations)):
        optimizer = optimizers[o]
        activation = activations[a]
        model = KerasClassifier(build_fn=build_model_2,verbose=1,batch_size=10,epochs=200,shuffle=False)
        kf = KFold(n_splits=5)
        result = cross_val_score(model,X,y,cv=kf)
        results_3.append(result)

# Print cross-validation score for each possible pair of optimizer, activation
c = 0
for o in range(len(optimizers)):
    for a in range(len(activations)):
        print("activation = ", activations[a],", optimizer = ", optimizers[o], ", Test Accuracy = ", abs(results_3[c].mean()))
        c += 1




