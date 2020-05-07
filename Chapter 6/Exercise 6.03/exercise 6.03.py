# -*- coding: utf-8 -*-
"""
Created on Wed May  6 17:04:26 2020

@author: PRAJWAL
"""

# Import the libraries
import numpy as np
import pandas as pd
# Load the Data
X = pd.read_csv("aps_failure_training_feats.csv")
y = pd.read_csv("aps_failure_training_target.csv")

from sklearn.model_selection import train_test_split
seed = 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=seed)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
# Transform the training data
X_train = sc.fit_transform(X_train)
X_train = pd.DataFrame(X_train,columns=X_test.columns)
# Transform the testing data
X_test = sc.transform(X_test)
X_test = pd.DataFrame(X_test,columns=X_train.columns)

# Import the relevant Keras libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from tensorflow import random
np.random.seed(seed)
random.set_seed(seed)
model = Sequential()
# Add the hidden dense layers and with dropout Layer
model.add(Dense(units=64, activation='relu', kernel_initializer='uniform', input_dim=X_train.shape[1]))
model.add(Dropout(rate=0.5))
model.add(Dense(units=32, activation='relu', kernel_initializer='uniform'))
model.add(Dropout(rate=0.4))
model.add(Dense(units=16, activation='relu', kernel_initializer='uniform'))
model.add(Dropout(rate=0.3))
model.add(Dense(units=8, activation='relu', kernel_initializer='uniform'))
model.add(Dropout(rate=0.2))
model.add(Dense(units=4, activation='relu', kernel_initializer='uniform'))
model.add(Dropout(rate=0.1))
# Add Output Dense Layer
model.add(Dense(units=1, activation='sigmoid', kernel_initializer='uniform'))
# Compile the Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


model.fit(X_train, y_train, epochs=100, batch_size=20, verbose=1, validation_split=0.2, shuffle=False)

y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)

from sklearn.metrics import confusion_matrix
y_pred_class1 = y_pred > 0.5
cm = confusion_matrix(y_test, y_pred_class1)
print(cm)

# True Negative
TN = cm[0,0]
# False Negative
FN = cm[1,0]
# False Positives
FP = cm[0,1]
# True Positives
TP = cm[1,1]

# Calculating Sensitivity
Sensitivity = TP / (TP + FN)
print(f'Sensitivity: {Sensitivity:.4f}')

# Calculating Specificity
Specificity = TN / (TN + FP)
print(f'Specificity: {Specificity:.4f}')

# Precision
Precision = TP / (TP + FP)
print(f'Precision: {Precision:.4f}')

# Calculate False positive rate
False_Positive_rate = FP / (FP + TN)
print(f'False positive rate: {False_Positive_rate:.4f}')

y_pred_class2 = y_pred > 0.3

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred_class2)
print(cm)

# True Negative
TN = cm[0,0]
# False Negative
FN = cm[1,0]
# False Positives
FP = cm[0,1]
# True Positives
TP = cm[1,1]

# Calculating Sensitivity
Sensitivity = TP / (TP + FN)
print(f'Sensitivity: {Sensitivity:.4f}')

# Calculating Specificity
Specificity = TN / (TN + FP)
print(f'Specificity: {Specificity:.4f}')
