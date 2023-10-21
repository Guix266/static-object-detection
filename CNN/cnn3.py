# -*- coding: utf-8 -*-

"""
Module used to train a CNN model from a pretrained ResNet50
"""


import tensorflow as tf
from sklearn.model_selection import train_test_split
import sklearn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import sys
import seaborn as sn
import cv2

import cnn2

from tensorflow.keras.applications.resnet50 import ResNet50, decode_predictions, preprocess_input


# =============================================================================
#  Import the classes :
# =============================================================================

CLASSES = ['backpack', 'suitcase', 'handbag', 'bench', 'chair', 'person', 
          'train', 'bicycle', 'motorcycle', 'bus', 'truck', 'car', 'fire hydrant' ]
NB_CLASSES = len(CLASSES)
im_size = (64, 64, 3)

(Xtrain, Ytrain) = cnn2.import_all_data("data_subsets\\train2017", CLASSES, im_size = im_size)
(Xval, Yval) = cnn2.import_all_data("data_subsets\\val2017", CLASSES, im_size = im_size)

# =============================================================================
#  Data preprocessing
# =============================================================================
#%%

def repartition(Y):
    unique_elements, counts_elements = np.unique(Y, return_counts=True)
    x = unique_elements
    plt.bar(x, counts_elements)
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.title("Original dataset distribution")
    plt.show()
    
def downsampling(X, Y):
    """Downsample the dataset"""
    # Get the minimum number class
    unique_elements, counts_elements = np.unique(Y, return_counts=True)
    class_min = unique_elements[0]
    mini = counts_elements[0]
    for i in range(len(unique_elements)):
        if counts_elements[i] <= mini:
            class_min = unique_elements[i]
            mini = counts_elements[i]
            
    # delete the elements that are too numerous
    t, n, m, c = np.shape(X)
    new_X = np.zeros((mini*np.shape(unique_elements)[0], n, m, c))
    new_Y = np.zeros((mini*np.shape(unique_elements)[0]))
    count = np.zeros(np.shape(unique_elements))
    # print(np.shape(new_X))
    j = 0
    for i in range(len(X)):
        if count[Y[i]] < mini:
            new_X[j,:,:,:] = X[i,:,:,:]
            new_Y[j] = Y[i]
            count[Y[i]] += 1
            j+=1
    return(new_X, new_Y)

# randomize
X = np.concatenate((Xtrain, Xval), axis=0)
y = np.concatenate((Ytrain, Yval), axis=0)
X, y = sklearn.utils.shuffle(X, y)

# Downsampling
# X, y = downsampling(X, y)

# Shuffle
# Xtrain, Ytrain = sklearn.utils.shuffle(Xtrain, Ytrain)
# Xval, Yval = sklearn.utils.shuffle(Xval, Yval)

# Preprocess the images
print("preprocessing of input.....")
Xtrain = preprocess_input(Xtrain)
Xval = preprocess_input(Xval)

repartition(Ytrain)
repartition(Yval)

# Split
Xtrain, Xval, Ytrain, Yval = train_test_split(X,y, test_size=0.2, random_state=0)
Ytrain = tf.keras.utils.to_categorical(Ytrain, NB_CLASSES, np.int32).astype(np.int8)
Yval = tf.keras.utils.to_categorical(Yval, NB_CLASSES, np.int32).astype(np.int8)



# Data augmentation
# create data generator
datagen = tf.keras.preprocessing.image.ImageDataGenerator(   
                        rotation_range=45,
                        width_shift_range=.15,
                        height_shift_range=.15,
                        horizontal_flip=True,
                        zoom_range=0.5)

datagen.fit(Xtrain)



# =============================================================================
#  Import the RESNET Model
# =============================================================================
#%%

print("[INFO] : Based model importation")
# Create the base model from the pre-trained model
base_model = ResNet50(input_shape=im_size, include_top=False, weights='imagenet')
print("[INFO] : Model Imported")

base_model.trainable = False

base_model.summary()
# =============================================================================
#   Build the new model
# =============================================================================
#%%

model = tf.keras.Sequential([ base_model,
                              tf.keras.layers.GlobalAveragePooling2D(),
                              tf.keras.layers.Dense(13, activation='softmax')
                              ])

# We can allow some of the resnet layers to change as we train.  
# Typically you would want to lower the learning rate in conjunction with this.
model.layers[0].trainable = True
# We let the last 2 blocks train
for layer in model.layers[0].layers[:-11]:
    layer.trainable = False
for layer in model.layers[0].layers[-11:]:
    layer.trainable = True


model.compile( loss = 'categorical_crossentropy',
                optimizer = 'Adam',
                metrics = ['accuracy'])

model.summary()

# Evaluate the inital performance of the model
loss0,accuracy0 = model.evaluate(Xval, Yval, batch_size = 256)
print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))


# =============================================================================
# # Train the model
# =============================================================================
#%%
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath="models/checkpoint/cp.ckpt",
                                                     save_weights_only=True,
                                                     monitor='val_accuracy', 
                                                     save_best_only=True,
                                                     verbose=1)

earlystop_callback = tf.keras.callbacks.EarlyStopping(
                    monitor='val_accuracy', min_delta=0.0001,
                    patience=5)

history = model.fit(datagen.flow(Xtrain, Ytrain, batch_size=32),
                    epochs = 10,
                    validation_data=(Xval, Yval))

# history = model.fit_generator(datagen.flow(Xtrain, Ytrain, batch_size=32),
#                     steps_per_epoch=len(Xtrain) / 32, 
#                     epochs=10,
#                     verbose = 1,
#                     callbacks=[earlystop_callback])

model.save('models/gui/pretrain_cnn.h5')

# Plot curves
cnn2.plot_curves(history)
    