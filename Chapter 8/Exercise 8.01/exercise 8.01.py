# -*- coding: utf-8 -*-
"""
Created on Mon May  4 23:02:28 2020

@author: PRAJWAL
"""

import numpy as np
from keras.applications.vgg16 import VGG16,preprocess_input
from keras.preprocessing import image 

classifier = VGG16()

new_image = image.load_img('../Data/Prediction/pizza.jpg', target_size=(224,224))
new_image


transformed_image = np.expand_dims(transformed_image,axis = 0)
transformed_image.shape

transformed_image = preprocess_input(transformed_image)
transformed_image

y_pred = classifier.predic(transformed_image)
y_pred

y_pred.shape

from keras.applications.vgg16 import decode_predictions
decode_predictions(y_pred,top=5)

label = decode_predictions(y_pred)

decoded_label = label[0][0]

print('%s (%.2f%%)' % (decoded_label[1], decoded_label[2]*100 ))

 
