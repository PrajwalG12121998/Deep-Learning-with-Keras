# -*- coding: utf-8 -*-
"""
Created on Mon May  4 15:39:01 2020

@author: PRAJWAL
"""

from keras.preprocessing import image
new_image = image.load_img('../test_image_1.jpg',target_size=(64,64))

training_set.class_indices

new_image = image.img_to_array(new_image)
new_image = np.expand_dims(new_image,axis = 0)


result = classifier.predict(new_image)

if result[0][0] == 1:
    prediction = 'It is a flower'
else:
    prediction = 'It is a car'

print(prediction)

