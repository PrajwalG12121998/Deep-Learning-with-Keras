# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 16:20:31 2020

@author: PRAJWAL
"""

import pandas as pd

X = pd.read_csv('tree_class_feats.csv')
y = pd.read_csv('tree_class_target.csv')

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)

from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np

model_1 = Sequential()
model_1.add(Dense(16, activation='relu', input_dim=10))
model_1.add(Dense(12, activation='relu'))
model_1.add(Dense(8, activation='relu'))
model_1.add(Dense(4, activation='relu'))
model_1.add(Dense(1, activation='sigmoid'))

model_1.compile(optimizer='sgd', loss='binary_crossentropy')

# train the model
history = model_1.fit(X_train, y_train, epochs=300, batch_size=50, verbose=1, shuffle=False)
# evaluate on test set
print("Test Loss =", model_1.evaluate(X_test, y_test))

#-------------------------------------------------------------------------------------------
#Model 2 with Dropout in first hidden layer

from keras.layers import Dropout

model_2 = Sequential()
model_2.add(Dense(16, activation='relu', input_dim=10))
model_2.add(Dropout(0.1))
model_2.add(Dense(12, activation='relu'))
model_2.add(Dense(8, activation='relu'))
model_2.add(Dense(4, activation='relu'))
model_2.add(Dense(1, activation='sigmoid'))

model_2.compile(optimizer='sgd', loss='binary_crossentropy')

# train the model
history = model_2.fit(X_train, y_train, epochs=300, batch_size=50, verbose=1, shuffle=False)
# evaluate on test set
print("Test Loss =", model_2.evaluate(X_test, y_test))

#-------------------------------------------------------------------------------------------
#Model 3 with Dropout in all hidden layer

model_3 = Sequential()
model_3.add(Dense(16, activation='relu', input_dim=10))
model_3.add(Dropout(0.2))
model_3.add(Dense(12, activation='relu'))
model_3.add(Dropout(0.2))
model_3.add(Dense(8, activation='relu'))
model_3.add(Dropout(0.2))
model_3.add(Dense(4, activation='relu'))
model_3.add(Dropout(0.2))
model_3.add(Dense(1, activation='sigmoid'))

model_3.compile(optimizer='sgd', loss='binary_crossentropy')

# train the model
history = model_3.fit(X_train, y_train, epochs=300, batch_size=50, verbose=1, shuffle=False)
# evaluate on test set
print("Test Loss =", model_3.evaluate(X_test, y_test))
