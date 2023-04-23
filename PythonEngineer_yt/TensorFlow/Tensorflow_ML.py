# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 20:18:43 2022

@author: Ensyuan
"""

#============================================
#  03 Neural Network
#============================================

# First Neural Net
# Train, evaluate, and predict with the model

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt


# datasets
mnist = keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape)
print(x_train[0]) # 數字在0-255之間

# normalize: 0-255 --> 0-1
x_train, x_test = x_train / 255.0, x_test / 255.0

#------------------
# show pics

plt.figure(figsize=(20,10))
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(x_train[i])
plt.show()


plt.figure(figsize=(20,10))
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(x_test[i])
plt.show()


#------------
# model
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)), # to one dimension
    keras.layers.Dense(128, activation='relu'), # Full connected layer
    keras.layers.Dense(10) # classification
    ])

print(model.summary)


# Another way to build the Sequential model:
# model = keras.models.Sequential()
# model.add(keras.layer.Flatten(input_shape=(28, 28)))
# model.add(keras.layers.Dense(128, activation='relu'))
# model.add(keras.layers.Dense(10))

#-------------------------
# loss and optimizer
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True) # 如果有用softmax, 好像就是false
optim = keras.optimizers.Adam(lr=0.001) # CityGrandma 的打法不用設定 learning rate, default=?
metrics = ['accuracy']


model.compile(loss=loss, optimizer=optim, metrics=metrics)

#-----------------
# training
batch_size = 64
epochs = 5

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=2)
                                                                # shuffle is highly recommended

#----------------
# evaluate
model.evaluate(x_test, y_test, batch_size=batch_size, verbose=2)


#-----------------
# prediction

# Option 1: build new model with Softmax layer

probability_model = keras.models.Sequential([
    model,
    keras.layers.Softmax() # 將 classification 換成機率，也就是這張圖片被歸在這一類的機率
    ])

predictions = probability_model(x_test)
pred0 = predictions[0]
print(pred0) # 第0張照片的prediction probability (to each class)
np.max(predictions[0])

# use np.argmax to get label with highest probability
label0 = np.argmax(pred0) # probability 最高的 class indax
print(label0)


# Option 2: original model + nn.softmax, call model(x)

predictions = model(x_test)
predictions = tf.nn.softmax(predictions)
pred1 = predictions[1]
print(pred1)
label1 = np.argmax(pred1)
print(label1)


# Option 3: original model + nn.softmax, call model.predict(x)

predictions = model.predict(x_test, batch_size=batch_size)
predictions = tf.nn.softmax(predictions)
pred2 = predictions[2]
print(pred2)
label2 = np.argmax(pred2)
print(label2)


#call argmax for multiple labels

pred05s = predictions[0:5]
print(pred05s.shape)
label05s = np.argmax(pred05s, axis=1)
print(label05s)


#============================================
#  04 Regression
#============================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Make numpy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing


# https://archive.ics.uci.edu/ml/datasets/Auto+MPG
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_name = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']

dataset = pd.read_csv(url)
dataset.tail()

dataset = pd.read_csv(url, names=column_name, na_values='?',
                      comment='\t', sep=' ', skipinitialspace=True)
dataset.tail()
dataset.info()
dataset.shape


#--------------------------------
# clean data

dataset = dataset.dropna()

# convert categorical 'Origin' data into one-hot data

origin = dataset.pop('Origin') # 'Origin' column is popped out
dataset['USA'] = (origin == 1)*1 
dataset['Europe'] = (origin == 2)*1
dataset['Japan'] = (origin == 3)*1

dataset.tail()


#------------------------------------------------
# Split the data into train and test

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

print(dataset.shape, train_dataset.shape, test_dataset.shape)
train_dataset.describe().transpose()


#--------------------------------------------------
# split features from labels

train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('MPG')
test_labels = test_features.pop('MPG')


def scaplot(feature, x=None, y=None):
    plt.figure(figsize=(10, 8))
    plt.scatter(train_features[feature], train_labels, label = 'Data')
    if x is not None and y is not None:
        plt.plot(x, y, color='k', label='Predictions')
    plt.xlabel(feature)
    plt.ylabel('MPG')
    plt.legend()

scaplot('Horsepower')
scaplot('Weight')


#---------------------------
# Normalization

print(train_dataset.describe().transpose()[['mean', 'std']])

normalizer = preprocessing.Normalization()

normalizer.adapt(np.array(train_features))
print(normalizer.mean.numpy())
    # 由此可知，data 有被標準化過


# When the layer is called it returns the input data, with each feature independently normalized:
# (input-mean)/stddev
first = np.array(train_features[:1])
print('First example:', first)
print('Normalized:', normalizer(first).numpy())


#------------------------------
# Set Model

# Regression
 # 1. Normalize the input horsepower
 # 2. Apply a linear transformation (y = m*x+b) to produce 1 output using layers.Dense

feature = 'Horsepower'
single_feature = np.array(train_features[feature])
print(single_feature.shape, train_features.shape)


# Normalization
single_feature_normalizer = preprocessing.Normalization()
single_feature_normalizer.adapt(single_feature)


# Sequentialmodel
single_feature_model = keras.models.Sequential([
    single_feature_normalizer,
    layers.Dense(units=1) # Linear Model
    ])

single_feature_model.summary()

#-----------------------------
# loss and optimizer

loss = keras.losses.MeanAbsoluteError() # MeanSquareError
optim = keras.optimizers.Adam(lr=0.1)

single_feature_model.compile(optimizer=optim, loss=loss)

#--------------------------
# history data

history = single_feature_model.fit(
    train_features[feature], train_labels,
    epochs=100,
    verbose=1,
    # calculate validation results on 20% of the training data
    validation_split = 0.2
    )

#-------------------
# plot function

def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0,25])
    plt.xlabel('Epoch')
    plt.ylabel('Error [MPG]')
    plt.legend()
    plt.grid(True)
plot_loss(history)

#-----------------------
# evaluate

single_feature_model.evaluate(
    test_features[feature],
    test_labels, verbose=1
    )

#-------------------------
# predict and plot

range_min = np.min(test_features[feature]) - 10
range_max = np.max(test_features[feature]) + 10
x = tf.linspace(range_min, range_max, 200)
y = single_feature_model.predict(x)

scaplot(feature, x, y)


#-------------------------
# DNN

dnn_model = keras.Sequential([
    single_feature_normalizer,
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
    ]) 

dnn_model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(0.001))

dnn_model.summary()

dnn_model.fit(
    train_features[feature], train_labels,
    validation_split=0.2,
    verbose=1, epochs=100
    )

dnn_model.evaluate(test_features[feature], test_labels, verbose=1)


#--------------------------------
# predict and plot

x = tf.linspace(range_min, range_max, 200)
y = dnn_model.predict(x)

scaplot(feature, x, y)


#---------------------------------
# multiple inputs

linear_model = tf.keras.Sequential([
    normalizer,
    layers.Dense(units=1)
])

linear_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss=loss)


linear_model.fit(
    train_features, train_labels, 
    epochs=100,
    verbose=1,
    # Calculate validation results on 20% of the training data
    validation_split = 0.2)


linear_model.evaluate(
    test_features, test_labels, verbose=1)


#--------------------------------
# predict and plot

x = tf.linspace(range_min, range_max, 200)
y = linear_model.predict(x)

scaplot('Weight', x, y)
scaplot('Horsepower', x, y)
scaplot('Cylinders', x, y)
scaplot('Displacement', x, y)
scaplot('Acceleration', x, y)
scaplot('Model Year', x, y)


column_name = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']


#============================================
#  05 CNN
#============================================

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt

#------------------------
# Dataset

cifar10 = keras.datasets.cifar10

(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

print(train_images.shape)


#---------------------------------
# Normalize: 0-255 --> 0-1
train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

def show():
    plt.figure(figsize=(10,10))
    for i in range(16):
        plt.subplot(4,4,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[train_labels[i][0]])
        # The CIFAR labels happen to be arrays, 
        # which is why you need the extra index
    plt.show()

show()


#---------------------------
# Set Model

model = keras.models.Sequential()
model.add(layers.Conv2D(32, (3,3), strides=(1,1), padding="valid", activation='relu', input_shape=(32,32,3)))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(32, 3, activation='relu'))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))
print(model.summary())

# import sys; sys.exit()

#--------------------------------
# loss and optimizer
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optim = keras.optimizers.Adam(lr=0.001)
metrics = ['accuracy']


model.compile(optimizer=optim, loss=loss, metrics=metrics)

# training
batch_size = 64
epochs=50

model.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs, verbose=2)


# evaluate
model.evaluate(test_images, test_labels, batch_size=batch_size, verbose=2)


#============================================
#  06 Save Load
#============================================

#------------------------
# 03 MNIST model

import tensorflow as tf
from tensorflow import keras
import numpy as np

mnist = keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# normalize
x_train, x_test = x_train / 255.0, x_test / 255.0


# Feed forward neural network
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10),
])

# config
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optim = keras.optimizers.Adam(lr=0.001) # "adam"
metrics = [keras.metrics.SparseCategoricalAccuracy()] # "accuracy"

# compile
model.compile(loss=loss, optimizer=optim, metrics=metrics)

# fit/training
model.fit(x_train, y_train, batch_size=64, epochs=5, shuffle=True, verbose=2)

print("Evaluate:")
model.evaluate(x_test,  y_test, verbose=2)


#-----------------------------------
# Save model

# 1) Save whole model
# two formats: SavedModel or HDF5

model.save('nn') # no file ending = SavedModel
model.save('nn.h5') # .h5 = HDF5

new_model = keras.models.load_model('nn.h5')


# 2) save only weights
model_save_weights('nn_weights.h5')

# initialize model first:
# model=keras.Sequential([])
model.load_weights('nn_weights.h5')


# 3) save only architecture, to_json
json_string = model.to_json()

with open('nn_model.json', 'w') as f:
    f.write(json_string)
    
with open('nn_model.json', 'r') as f:
    load_json_string = f.read()


new_model = keras.models.model_from_json(loaded_json_string)
print(new_model.summary())








































