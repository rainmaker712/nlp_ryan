#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 15:20:50 2017

@author: ryan

Most of infomation from DataCamp Keras Course
https://www.datacamp.com/community/blog/new-course-deep-learning-in-python-first-keras-2-0-online-course#gs.8RUVmWM

"""

# Import necessary modules
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical

# Convert the target to categorical: target
target = to_categorical(df.survived)

model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(n_cols,)))
model.add(Dense(2, activation='softmax'))
# Compile the model
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
# Fit the model
model.fit(predictors, target)

'''Predictions'''
# Calculate predictions: predictions
predictions = model.predict(pred_data)

# Calculate predicted probability of survival: predicted_prob_true
predicted_prob_true = predictions[:,1]

# print predicted_prob_true
print(predicted_prob_true)

'''
Save and Load
'''

from keras.models import load_model
model.save('model_file.h5')
my_model = load_model('my_model.h5')




