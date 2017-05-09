#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 18:27:14 2017

@author: ryan
"""


"""
Model Optimization

1. loss options
- mean_squared_error
- mean_squared_lograithmic_error
- mean_absolute_error
- mean_ablsolute_percentage_error
- binary_crossentropy
- categorical_crossentropy

2. L1/L2 regeularization

from keras import regularizers
model.add(Dense(50, input_dim=100, activation="sigmoid", W_regularizer=regularizers.l2(0.01)))

3. Dropout -> 마지막에 가중치 p를 곱하여 스케일링

model.add(Dropout(0.5))
model.compile(optimizer=SGD(0.5), loss='categorical_crossentropy', metrics=["acc"])

4. Weight initialization
model.add(Dense(100, input_dim=10, activation="sigmoid", "init"=uniform))

5. Softmax

model.Sequential()
model.add(Dense(15, input_dim=100, activation='sigmoid', init="global_uniform"))
model.add(Dense(10, activation='softmax', init='global_uniform"))
model.compile(optimizer=SGD(), loss='categorical_crossentropy', metrics=["accuracy"])

"""

# Import the SGD optimizer
from keras.optimizers import SGD

# Create list of learning rates: lr_to_test
lr_to_test = [.000001, 0.01, 1]

# Loop over learning rates
for lr in lr_to_test:
    print('\n\nTesting model with learning rate: %f\n'%lr )
    
    # Build new model to test, unaffected by previous models
    model = get_new_model()
    
    # Create SGD optimizer with specified learning rate: my_optimizer
    my_optimizer = SGD(lr=lr)
    
    # Compile the model
    model.compile(optimizer = my_optimizer, loss = 'categorical_crossentropy')
    
    # Fit the model
    model.fit(predictors, target)
    
    

    
    
    
    
"""
Model validation


model.fit(predictors, target, validation_split=0.3)
Early Stopping
stop traiing if validation is same (patient)

Experimentation
- Experiment with different architectures
- More layers
- Fewer layers
- Layers with more nodes
- Layers with fewer nodes
- Creating a great model requires experimentation

""""
#Validation Set
# Save the number of columns in predictors: n_cols
n_cols = predictors.shape[1]
input_shape = (n_cols,)

# Specify the model
model = Sequential()
model.add(Dense(100, activation='relu', input_shape = input_shape))
model.add(Dense(100, activation='relu'))
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])

# Fit the model
hist = model.fit(predictors, target, validation_split=0.3)

"""
#Early Stopping
"""

# Import EarlyStopping
from keras.callbacks import EarlyStopping

# Save the number of columns in predictors: n_cols
n_cols = predictors.shape[1]
input_shape = (n_cols,)

# Specify the model
model = Sequential()
model.add(Dense(100, activation='relu', input_shape = input_shape))
model.add(Dense(100, activation='relu'))
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define early_stopping_monitor
early_stopping_monitor = EarlyStopping(patience = 2)

# Fit the model
model.fit(predictors, target, epochs=30, validation_split=0.3, callbacks = [early_stopping_monitor])

"""
##Experimenting with wider networks

verbose=False / logging output, tell me everything

"""
# Define early_stopping_monitor
early_stopping_monitor = EarlyStopping(patience=2)

# Create the new model: model_2
model_2 = Sequential()

# Add the first and second layers
model_2.add(Dense(100, activation="relu", input_shape=input_shape))
model_2.add(Dense(100, activation="relu"))

# Add the output layer
model_2.add(Dense(2, activation="softmax"))

# Compile model_2
model_2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Fit model_1
model_1_training = model_1.fit(predictors, target, epochs=15, validation_split=0.2, callbacks=[early_stopping_monitor], verbose=False)

# Fit model_2
model_2_training = model_2.fit(predictors, target, epochs=15, validation_split=0.2, callbacks=[early_stopping_monitor], verbose=False)

# Create the plot
plt.plot(model_1_training.history['val_loss'], 'r', model_2_training.history['val_loss'], 'b')
plt.xlabel('Epochs')
plt.ylabel('Validation score')
plt.show()



