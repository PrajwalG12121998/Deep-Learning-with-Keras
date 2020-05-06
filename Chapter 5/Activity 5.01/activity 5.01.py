# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 15:30:47 2020

@author: PRAJWAL
"""

import pandas as pd

X = pd.read_csv('avila-tr_feats.csv')
y = pd.read_csv('avila-tr_target.csv')

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

from keras.models import Sequential
from keras.layers import Dense,Activation

model = Sequential()
model.add(Dense(10,activation='relu',input_dim=X.shape[1]))
model.add(Dense(6,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer='sgd',loss='binary_crossentropy',metrics=['accuracy'])

history = model.fit(X_train,y_train,epochs=100,batch_size=20,shuffle=False,validation_data=(X_test,y_test))

import matplotlib.pyplot as plt 
import matplotlib
# plot training error and test error
matplotlib.rcParams['figure.figsize'] = (10.0, 8.0) 
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylim(0,1)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train loss', 'validation loss'], loc='upper right')
# print the best accuracy reached on the test set
print("Best Accuracy on Validation Set =", max(history.history['val_accuracy']))

#--------------------------------------------------------------------------------------------------
#Model 2 with L2 regularization and lambda value 0.01

from keras.regularizers import l2
l2_param = 0.01

model_2 = Sequential()
model_2.add(Dense(10,activation='relu',input_dim=X.shape[1],kernel_regularizer=l2(l2_param)))
model_2.add(Dense(6,activation='relu',kernel_regularizer=l2(l2_param)))
model_2.add(Dense(4,activation='relu',kernel_regularizer=l2(l2_param)))
model_2.add(Dense(1,activation='sigmoid'))

model_2.compile(optimizer='sgd',loss='binary_crossentropy',metrics=['accuracy'])
 
history = model_2.fit(X_train,y_train,epochs=100,batch_size=20,shuffle=False,validation_data=(X_test,y_test))

matplotlib.rcParams['figure.figsize'] = (10.0, 8.0) 
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylim(0,1)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train loss', 'validation loss'], loc='upper right')
# print the best accuracy reached on the test set
print("Best Accuracy on Validation Set =", max(history.history['val_accuracy']))

#--------------------------------------------------------------------------------------------------
#Model 3 with L2 regularization and lambda value 0.1

l2_param = 0.1

model_3 = Sequential()
model_3.add(Dense(10,activation='relu',input_dim=X.shape[1],kernel_regularizer=l2(l2_param)))
model_3.add(Dense(6,activation='relu',kernel_regularizer=l2(l2_param)))
model_3.add(Dense(4,activation='relu',kernel_regularizer=l2(l2_param)))
model_3.add(Dense(1,activation='sigmoid'))

model_3.compile(optimizer='sgd',loss='binary_crossentropy',metrics=['accuracy'])
 
history = model_3.fit(X_train,y_train,epochs=100,batch_size=20,shuffle=False,validation_data=(X_test,y_test))

matplotlib.rcParams['figure.figsize'] = (10.0, 8.0) 
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylim(0,1)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train loss', 'validation loss'], loc='upper right')
# print the best accuracy reached on the test set
print("Best Accuracy on Validation Set =", max(history.history['val_accuracy']))

#--------------------------------------------------------------------------------------------------
#Model 4 with L2 regularization and lambda value 0.005

l2_param = 0.005

model_4 = Sequential()
model_4.add(Dense(10,activation='relu',input_dim=X.shape[1],kernel_regularizer=l2(l2_param)))
model_4.add(Dense(6,activation='relu',kernel_regularizer=l2(l2_param)))
model_4.add(Dense(4,activation='relu',kernel_regularizer=l2(l2_param)))
model_4.add(Dense(1,activation='sigmoid'))

model_4.compile(optimizer='sgd',loss='binary_crossentropy',metrics=['accuracy'])
 
history = model_4.fit(X_train,y_train,epochs=100,batch_size=20,shuffle=False,validation_data=(X_test,y_test))

matplotlib.rcParams['figure.figsize'] = (10.0, 8.0) 
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylim(0,1)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train loss', 'validation loss'], loc='upper right')
# print the best accuracy reached on the test set
print("Best Accuracy on Validation Set =", max(history.history['val_accuracy']))

#--------------------------------------------------------------------------------------------------
#Model 5 with L1 regularization and lambda value 0.01

from keras.regularizers import l1

l1_param = 0.01

model_5 = Sequential()
model_5.add(Dense(10,activation='relu',input_dim=X.shape[1],kernel_regularizer=l1(l1_param)))
model_5.add(Dense(6,activation='relu',kernel_regularizer=l1(l1_param)))
model_5.add(Dense(4,activation='relu',kernel_regularizer=l1(l1_param)))
model_5.add(Dense(1,activation='sigmoid'))

model_5.compile(optimizer='sgd',loss='binary_crossentropy',metrics=['accuracy'])
 
history = model_5.fit(X_train,y_train,epochs=100,batch_size=20,shuffle=True,validation_data=(X_test,y_test))

matplotlib.rcParams['figure.figsize'] = (10.0, 8.0) 
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylim(0,1)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train loss', 'validation loss'], loc='upper right')
# print the best accuracy reached on the test set
print("Best Accuracy on Validation Set =", max(history.history['val_accuracy']))

#--------------------------------------------------------------------------------------------------
#Model 6 with L1 regularization and L2 regularization and lambda value 0.005

from keras.regularizers import l1_l2
l1_param = 0.005
l2_param = 0.005

model_6 = Sequential()
model_6.add(Dense(10,activation='relu',input_dim=X.shape[1],kernel_regularizer=l1_l2(l1=l1_param,l2=l2_param)))
model_6.add(Dense(6,activation='relu',kernel_regularizer=l1_l2(l1=l1_param,l2=l2_param)))
model_6.add(Dense(4,activation='relu',kernel_regularizer=l1_l2(l1=l1_param,l2=l2_param)))
model_6.add(Dense(1,activation='sigmoid'))

model_5.compile(optimizer='sgd',loss='binary_crossentropy',metrics=['accuracy'])
 
history = model_5.fit(X_train,y_train,epochs=100,batch_size=20,shuffle=False,validation_data=(X_test,y_test))

matplotlib.rcParams['figure.figsize'] = (10.0, 8.0) 
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylim(0,1)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train loss', 'validation loss'], loc='upper right')
# print the best accuracy reached on the test set
print("Best Accuracy on Validation Set =", max(history.history['val_accuracy']))


