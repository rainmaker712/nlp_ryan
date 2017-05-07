#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 15:20:50 2017

@author: ryan

Most of infomation from DataCamp Keras Course
https://www.datacamp.com/community/blog/new-course-deep-learning-in-python-first-keras-2-0-online-course#gs.8RUVmWM

"""

# Import necessary modules
#import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.datasets import boston_housing
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import numpy as np

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

print(x_train.shape, y_train.shape) #(404, 13) / (404,)

# Save the number of columns in training set: n_cols
n_cols = x_train.shape[1]

#Define Model for boston data

# Set up the model: model
model = Sequential()
model.add(Dense(13, activation='relu', input_shape=(n_cols,), kernel_initializer = 'normal'))
# Add the output layer
model.add(Dense(1, kernel_initializer='normal'))
#Complile model 일반적으로 Adam을 추천 (CS231 강의에서도 잘 모르겠으면 Adam 사용 추천)
model.compile(optimizer='adam', loss='mean_squared_error')

# Verify that model contains information from compiling
print("Loss function: " + model.loss)

"""
모델 학습 / 구조 확인 및 시각화
"""
model.summary() #모델의 구조를 확인
# Fit the model
history = model.fit(x_train, y_train, epochs=100)
# Test the model
'''Predictions'''
# Calculate predictions: predictions
score = model.evaluate(x_test, y_test)

# list all data in history
print(history.history.keys())

#Loss 시각화
import matplotlib.pyplot as plt

plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()


