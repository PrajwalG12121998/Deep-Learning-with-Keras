# -*- coding: utf-8 -*-
"""
Created on Mon May  4 23:57:32 2020

@author: PRAJWAL
"""

import numpy as np
import keras
from keras.layers import Dense
from tensorflow import random

vgg_model = keras.applications.vgg16.VGG16()
vgg_model.summary()

last_layer = str(vgg_model.layers[-1])
np.random.seed(42)

classifier = keras.Sequential()

for layer in vgg_model:
    if str(layer)!= last_layer:
        classifier.add(layer)
        

classifier.summary()

for layer in classifier.layers:
    layer.trainable = False
    
classifier.add(Dense(1,activation='sigmoid'))
classifier.summary()


classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator
generate_train_data = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
generate_test_data = ImageDataGenerator(rescale =1./255)
training_dataset = generate_train_data.flow_from_directory('../Data/dataset/training_set',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'binary')
test_dataset = generate_test_data.flow_from_directory('../Data/dataset/test_set',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'binary')
classifier.fit_generator(training_dataset,
                         steps_per_epoch = 100,
                         epochs = 10,
                         validation_data = test_dataset,
                         validation_steps = 30,
                         shuffle=False)

from keras.preprocessing import image
new_image = image.load_img('../Data/Prediction/test_image_2.jpg', target_size = (224, 224))
new_image = image.img_to_array(new_image)
new_image = np.expand_dims(new_image, axis = 0)
result = classifier.predict(new_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'It is a flower'
else:
    prediction = 'It is a car'
print(prediction)
