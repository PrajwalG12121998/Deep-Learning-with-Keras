# -*- coding: utf-8 -*-
"""
Created on Mon May  4 13:31:58 2020

@author: PRAJWAL
"""

from keras.models import Sequential
from keras.layers import Dense,Activation,Conv2D,MaxPool2D,Flatten
import numpy as np
from tensorflow import random

seed = 1
np.random.seed(seed)

classifier = Sequential()
classifier.add(Conv2D(32,3,3, input_shape = (64,64,3),activation='relu'))
classifier.add(Conv2D(32, (3,3), activation='relu'))
classifier.add(Conv2D(32, (3,3), activation='relu'))

#Pooling layer
classifier.add(MaxPool2D(2,2))

classifier.add(Conv2D(32, (3,3), activation='relu'))
classifier.add(MaxPool2D(pool_size=(2,2)))

classifier.add(Flatten())

#Adding layers of ANN
classifier.add(Dense(128,activation='relu'))
classifier.add(Dense(128,activation='relu'))
classifier.add(Dense(128,activation='relu'))
classifier.add(Dense(128,activation='relu'))
classifier.add(Dense(128,activation='softmax'))


classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

from keras.preprocessing import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range=0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('../dataset/training_set',
target_size = (64, 64),
batch_size = 32,
class_mode = 'binary')

test_set = test_datagen.flow_from_directory('../dataset/test_set',
target_size = (64, 64),
batch_size = 32,
class_mode = 'binary')

classifier.fit_generator(training_set,
steps_per_epoch = 10000,
epochs = 2,
validation_data = test_set,
validation_steps = 2500,
shuffle=False)

