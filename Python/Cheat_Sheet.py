#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 19:26:51 2017

@author: ryan
"""

#-----------------Sklearn--------------------
#1. Divide train and test data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

#---------------Keras----------------

#Create the plot
import matplotlib.pyplot as plt
plt.plot(model['acc'], 'r')
plt.xlabel('Epochs')
plt.ylabel('acc')
plt.show()

#Save Model
from keras.models import load_model
model.save('domain_classify.h5')

#Load Model
my_model = load_model('domain_classify.h5')

#Use Model (Make sure input as same dim.)
my_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
my_model.predict_classes(np.array(sent).shape)


