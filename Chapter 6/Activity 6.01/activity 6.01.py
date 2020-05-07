# -*- coding: utf-8 -*-
"""
Created on Wed May  6 17:09:50 2020

@author: PRAJWAL
"""

# Import the libraries
import numpy as np
import pandas as pd
# Load the Data
X = pd.read_csv("aps_failure_training_feats.csv")
y = pd.read_csv("aps_failure_training_target.csv")
# Use the head function to get a glimpse data
X.head()

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
seed = 13
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

# Initialize StandardScaler
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
# Transform the training data
X_train = sc.fit_transform(X_train)
X_train = pd.DataFrame(X_train, columns=X_test.columns)
# Transform the testing data
X_test = sc.transform(X_test)
X_test = pd.DataFrame(X_test, columns = X_train.columns)

# Import the relevant Keras libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
From tensorflow import random

# Initiate the Model with Sequential Class
np.random.state(seed)
random.set_seed(seed)
model = Sequential()

# Add the hidden dense layers and with dropout Layer
model.add(Dense(units=64, activation='relu', kernel_initializer='uniform', input_dim=X_train.shape[1]))
model.add(Dropout(rate=0.5))
model.add(Dense(units=32, activation='relu', kernel_initializer='uniform', input_dim=X_train.shape[1]))
model.add(Dropout(rate=0.4))
model.add(Dense(units=16, activation='relu', kernel_initializer='uniform', input_dim=X_train.shape[1]))
model.add(Dropout(rate=0.3))
model.add(Dense(units=8, activation='relu', kernel_initializer='uniform', input_dim=X_train.shape[1]))
model.add(Dropout(rate=0.2))
model.add(Dense(units=4, activation='relu', kernel_initializer='uniform'))
model.add(Dropout(rate=0.1))


# Add Output Dense Layer
model.add(Dense(units=1, activation='sigmoid', kernel_initializer='uniform'))

# Compile the Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=100, batch_size=20, verbose=1, validation_split=0.2, shuffle=False)

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'The loss on the test set is {test_loss:.4f} and the accuracy is {test_acc*100:.4f}%')



