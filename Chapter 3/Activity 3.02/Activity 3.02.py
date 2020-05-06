# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 17:56:15 2020

@author: PRAJWAL
"""

#Prediction of Advanced Fibrosis
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Activation
import matplotlib.pyplot as plt

X = pd.read_csv('HCV_feats.csv')
y = pd.read_csv('HCV_target.csv')

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scale = scaler.fit_transform(X)


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_scale,y,test_size=0.2,random_state=42)

#Model creation
model = Sequential()
model.add(Dense(3,activation='tanh',input_dim=28))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='sgd',loss='binary_crossentropy',metrics=['accuracy'])

history = model.fit(X_train,y_train,batch_size=20, epochs=100, validation_data=(X_test,y_test), shuffle=False)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Train','Test'],loc='upper left')
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Train','Test'],loc='upper left')
plt.show()

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'The loss on the test set is {test_loss:.4f} and the accuracy is {test_acc*100:.3f}%')

#Print important values
print('min_train_loss=',min(history.history['loss']))
print('min_test_loss=',min(history.history['val_loss']))
print('max_train_acc=',max(history.history['accuracy']))
print('max_test_acc=',max(history.history['val_accuracy']))

#------------------------------------------------------------------------------------------
#Model 2

model_2 = Sequential()
model_2.add(Dense(4,activation='tanh',input_dim=28))
model_2.add(Dense(3,activation='tanh'))
model_2.add(Dense(1,activation='sigmoid'))
model_2.compile(optimizer='sgd',loss='binary_crossentropy',metrics=['accuracy'])

history = model_2.fit(X_train,y_train,batch_size=20, epochs=100, validation_split=0.1, shuffle=False)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Train','Test'],loc='upper left')
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Train','Test'],loc='upper left')
plt.show()

#Print important values
print('min_train_loss=',min(history.history['loss']))
print('min_test_loss=',min(history.history['val_loss']))
print('max_train_acc=',max(history.history['accuracy']))
print('max_test_acc=',max(history.history['val_accuracy']))

test_loss_2, test_acc_2 = model_2.evaluate(X_test, y_test['AdvancedFibrosis'])
print(f'The loss on the test set is {test_loss_2:.4f} and the accuracy is {test_acc_2*100:.3f}%')


