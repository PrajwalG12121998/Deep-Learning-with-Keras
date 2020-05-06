# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 00:22:44 2020

@author: PRAJWAL
"""

import pandas as pd

X = pd.read_csv('tree_class_feats.csv')
y = pd.read_csv('tree_class_target.csv')

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)

from keras.models import Sequential
from keras.layers import Dense,Activation

model_1 = Sequential()
model_1.add(Dense(16,activation='relu',input_dim=X.shape[1]))
model_1.add(Dense(8,activation='relu'))
model_1.add(Dense(4,activation='relu'))
model_1.add(Dense(1,activation='sigmoid'))

model_1.compile(optimizer='sgd',loss='binary_crossentropy')

history = model_1.fit(X_train,y_train,epochs=300,batch_size=50,shuffle=False,validation_data=(X_test,y_test))

import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['figure.figsize'] = (10.0, 8.0)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylim(0,1)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train loss', 'validation loss'], loc='upper right')


#--------------------------------------------------------------------------------------------------------------------------------
#Model 2 with callbacks (Early stopping)
from keras.callbacks import EarlyStopping

model_2 = Sequential()
model_2.add(Dense(16,activation='relu',input_dim=X.shape[1]))
model_2.add(Dense(8,activation='relu'))
model_2.add(Dense(4,activation='relu'))
model_2.add(Dense(1,activation='sigmoid'))

model_2.compile(optimizer='sgd',loss='binary_crossentropy')

es_callback = EarlyStopping(monitor='val_loss',mode='min')
#train the model
history = model_2.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=300,batch_size=50,callbacks=[es_callback],shuffle=False)

matplotlib.rcParams['figure.figsize'] = (10.0, 8.0)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train loss', 'validation loss'], loc='upper right')


#--------------------------------------------------------------------------------------------------------------------------------
#Model 3 with callbacks (Early stopping)
from keras.callbacks import EarlyStopping

model_3 = Sequential()
model_3.add(Dense(16,activation='relu',input_dim=X.shape[1]))
model_3.add(Dense(8,activation='relu'))
model_3.add(Dense(4,activation='relu'))
model_3.add(Dense(1,activation='sigmoid'))

model_3.compile(optimizer='sgd',loss='binary_crossentropy')

es_callback = EarlyStopping(monitor='val_loss',mode='min',patience=10)
#train the model
history = model_3.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=300,batch_size=50,callbacks=[es_callback],shuffle=False)

matplotlib.rcParams['figure.figsize'] = (10.0, 8.0)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylim(0,1)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train loss', 'validation loss'], loc='upper right')



