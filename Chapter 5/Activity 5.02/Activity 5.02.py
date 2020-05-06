# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 16:43:48 2020

@author: PRAJWAL
"""

import pandas as pd

X = pd.read_csv('traffic_volume_feats.csv')
y = pd.read_csv('traffic_volume_target.csv')


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)

from keras.models import Sequential
from keras.layers import Dense,Activation

model_1 = Sequential()
model_1.add(Dense(10,activation='relu',input_dim=X.shape[1]))
model_1.add(Dense(10,activation='relu'))
model_1.add(Dense(1))

model_1.compile(optimizer='rmsprop',loss='mean_squared_error')

history = model_1.fit(X_train,y_train,epochs=200,batch_size=50,shuffle=False,validation_data=(X_test,y_test))

import matplotlib.pyplot as plt 
import matplotlib
 
matplotlib.rcParams['figure.figsize'] = (10.0, 8.0) 
# plot training error and test error plots 
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylim((0, 100))
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train loss', 'validation loss'], loc='upper right')
# print the best accuracy reached on the test set
print("Lowest error on training set = ", min(history.history['loss']))
print("Lowest error on validation set = ", min(history.history['val_loss']))

#----------------------------------------------------------------------------------------------------------
#Model 2 with Dropout regularization
from keras.layers import Dropout

model_2 = Sequential()
model_2.add(Dense(10,activation='relu',input_dim=X.shape[1]))
model_2.add(Dropout(0.01))
model_2.add(Dense(10,activation='relu'))
model_2.add(Dense(1))

model_2.compile(optimizer='rmsprop',loss='mean_squared_error')

history = model_2.fit(X_train,y_train,epochs=200,batch_size=50,shuffle=False,validation_data=(X_test,y_test))


 
matplotlib.rcParams['figure.figsize'] = (10.0, 8.0) 
# plot training error and test error plots 
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylim((0, 100))
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train loss', 'validation loss'], loc='upper right')
# print the best accuracy reached on the test set
print("Lowest error on training set = ", min(history.history['loss']))
print("Lowest error on validation set = ", min(history.history['val_loss']))

#--------------------------------------------------------------------------------------------------------------------
#Model 3 with Dropout regularization

model_3 = Sequential()
model_3.add(Dense(10,activation='relu',input_dim=X.shape[1]))
model_3.add(Dropout(0.01))
model_3.add(Dense(10,activation='relu'))
model_3.add(Dropout(0.01))
model_3.add(Dense(1))

model_3.compile(optimizer='rmsprop',loss='mean_squared_error')

history = model_3.fit(X_train,y_train,epochs=200,batch_size=50,shuffle=False,validation_data=(X_test,y_test))


 
matplotlib.rcParams['figure.figsize'] = (10.0, 8.0) 
# plot training error and test error plots 
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train loss', 'validation loss'], loc='upper right')
# print the best accuracy reached on the test set
print("Lowest error on training set = ", min(history.history['loss']))
print("Lowest error on validation set = ", min(history.history['val_loss']))

#--------------------------------------------------------------------------------------------------------------------
#Model 4 with Dropout regularization

model_4 = Sequential()
model_4.add(Dense(10,activation='relu',input_dim=X.shape[1]))
model_4.add(Dropout(0.02))
model_4.add(Dense(10,activation='relu'))
model_4.add(Dropout(0.01))
model_4.add(Dense(1))

model_4.compile(optimizer='rmsprop',loss='mean_squared_error')

history = model_4.fit(X_train,y_train,epochs=200,batch_size=50,shuffle=False,validation_data=(X_test,y_test))


 
matplotlib.rcParams['figure.figsize'] = (10.0, 8.0) 
# plot training error and test error plots 
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train loss', 'validation loss'], loc='upper right')
# print the best accuracy reached on the test set
print("Lowest error on training set = ", min(history.history['loss']))
print("Lowest error on validation set = ", min(history.history['val_loss']))








