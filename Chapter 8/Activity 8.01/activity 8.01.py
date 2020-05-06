# -*- coding: utf-8 -*-
"""
Created on Mon May  4 23:22:16 2020

@author: PRAJWAL
"""

import numpy as np
from keras.applications.vgg16 import VGG16,preprocess_input
from keras.preprocessing import image

classifier = VGG16()
classifier.summary()

new_image = image.load_img('../Data/Prediction/test_image_1.jpg', target_size=(224, 224))
new_image

transformed_image = image.img_to_array(new_image)
transformed_image.shape

transformed_image = np.expand_dims(transformed_image, axis=0)
transformed_image.shape

transformed_image = preprocess_input(transformed_image)
transformed_image

y_pred = classifier.predict(transformed_image)
y_pred

from keras.applications.vgg16 import decode_predictions
decode_predictions(y_pred, top=5)

label = decode_predictions(y_pred)
# Most likely result is retrieved, for example, the highest probability
decoded_label = label[0][0]
# The classification is printed
print('%s (%.2f%%)' % (decoded_label[1], decoded_label[2]*100 ))

