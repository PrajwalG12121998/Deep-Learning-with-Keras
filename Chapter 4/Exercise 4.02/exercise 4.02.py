# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 12:14:02 2020

@author: PRAJWAL
"""

# import data
import pandas as pd
colnames = ['CIC0', 'SM1_Dz(Z)', 'GATS1i', 'NdsCH', 'NdssC','MLOGP', 'LC50']
dataset = pd.read_csv('qsar_fish_toxicity.csv', sep=';', names=colnames)
X = dataset.drop('LC50', axis=1)
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


from keras.wrappers.scikit_learn import KerasRegressor
import numpy as np
from tensorflow import random
seed = 1
np.random.seed(seed)
#random.set_seed(seed)

YourModel = KerasRegressor(build_fn=build_model,verbose=1,batch_size=20,epochs=100,shuffle=False)

#5 fold cross-validation
from sklearn.model_selection import KFold
kf = KFold(n_splits=5)

from sklearn.model_selection import cross_val_score
results = cross_val_score(YourModel,X,y,cv=kf)
 
# print the result
print(f"Final Cross-Validation Loss = {abs(results.mean()):.4f}") 


