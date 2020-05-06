# -*- coding: utf-8 -*-
"""
Created on Tue May  5 21:06:07 2020

@author: PRAJWAL
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow import random

dataset_training = pd.read_csv('../AMZN_train.csv') 

training_data = dataset_training[['Open']].values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
training_data_scaled = sc.fit_transform(training_data)

X_train = []
y_train = []

for i in range(60,1258):
    X_train.append(training_data_scaled[i-60:i,0])
    y_train.append(training_data_scaled[i,0])
    
X_train,y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1],1))

from keras.models import Sequential
from keras.layers import Dense,LSTM,Dropout


seed = 1
np.random.seed(seed)
random.set_random_seed(seed)
model = Sequential()

model.add(LSTM(units=50, return_sequences=True,input_shape = (X_train.shape[1],1)))

#Adding second LSTM layer
model.add(LSTM(units=50,return_sequences=True))
#Adding third LSTM layer
model.add(LSTM(units=50,return_sequences=True))
#Adding fourth LSTM layer
model.add(LSTM(units=50))
#Adding the output layer
model.add(Dense(units=1))



#Compiling RNN
model.compile(optimizer='adam',loss = 'mean_squared_error')
# Fitting the RNN to the Training set
model.fit(X_train, y_train, epochs = 100, batch_size = 32)

dataset_testing = pd.read_csv('../AMZN_test.csv')
actual_stock_price = dataset_testing[['Open']].values

total_data = pd.concat((dataset_training['Open'], dataset_testing['Open']), axis = 0)

inputs = total_data[len(total_data) - len(dataset_testing) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 81):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)


# Visualizing the results
plt.plot(actual_stock_price, color = 'green', label = 'Real Amazon Stock Price',ls='--')
plt.plot(predicted_stock_price, color = 'red', label = 'Predicted Amazon Stock Price',ls='-')
plt.title('Predicted Stock Price')
plt.xlabel('Time in days')
plt.ylabel('Real Stock Price')
plt.legend()
plt.show()


