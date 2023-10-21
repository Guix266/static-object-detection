# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 12:14:09 2020

@author: guix
"""
import tensorflow as tf

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image, ImageOps



# =============================================================================
#     Functions
# =============================================================================

def import_data(folder_name, classes, max_images, im_size, grey) :
    class_size = []
    for category in classes:    
        class_size.append(min(max_images, len(os.listdir(os.path.join(folder_name, category))) -1))
    size = sum(class_size)
    if grey == True :
        X = np.zeros((size, im_size[0], im_size[1]))
    else :
        X = np.zeros((size, im_size[0], im_size[1],  im_size[2]))
    Y = np.zeros(size, dtype = np.uint8)
    permutation = np.random.permutation(size)
    k = 0
    for j in range(len(classes)):
        print("[INFO] : Import class " + classes[j], end='\r')
        for i in range(class_size[j]):
            num = str(i)
            while len(num) < 5:
                num = "0"+num
            if grey :     # grey
                # print(os.path.join(folder_name +"/" + classes[j], classes[j] + "_" + num +".jpg"))
                im = cv2.imread(os.path.join(folder_name +"/" + classes[j], classes[j] + "_" + num +".jpg"), cv2.IMREAD_GRAYSCALE)
                # plt.imshow(im,  cmap = 'gray')
                # plt.show()
                # X[k] = im / 255
                X[permutation[k]] = normalize(im)
                # X[permutation[k]] = im
            else :        # RGB
                # im = cv2.imread(os.path.join(folder_name +"/" + classes[j], classes[j] + "_" + num +".jpg"), cv2.IMREAD_COLOR)
                im = cv2.imread(os.path.join(folder_name +"/" + classes[j], classes[j] + "_" + num +".jpg"), cv2.IMREAD_COLOR)
                # im = cv2.resize(im, (96,96))
                # X[k] = normalize(im)
                # im = tf.cast(im, tf.float32)  
                X[permutation[k]] = (im/127.5) - 1
            Y[permutation[k]] = j
            k += 1
    return (X, Y)

def normalize(image_array) :
    mean, std = image_array.mean(), image_array.std()
    return  (image_array - mean)/std

def create_model(im_size):
    """creation of the CNN model with 2 convolutions layers"""
    
    model = Sequential([
        Conv2D(64, (3,3),padding='same', activation='relu',input_shape=im_size),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.2),
        Conv2D(32, (3, 3),padding='same', activation='relu'),
        Conv2D(32, (3, 3),padding='same', activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(NB_CLASSES, activation='softmax')
    ])

    #Compilation of the model
    model.compile(  loss = 'sparse_categorical_crossentropy',
                    optimizer ='Adam',
                    metrics = ['accuracy'])
    return(model)

def Training(model, trainX, trainY, testX, testY):
    """Train the input model with 25 epochs"""
    print("[INFO] : Training of the CNN ...")

    history = model.fit(trainX, 
                        trainY,
                        batch_size= 100,
                        epochs=5,
                        validation_data=(valX, valY),
                        verbose = 1,
                        )
    return(model, history)

def plot_curves(history):
    """Trace learning curve/accuracy rate"""
    # Get the values
    loss_curve = history.history["loss"]
    acc_curve = history.history["accuracy"]
    loss_val_curve = history.history["val_loss"]
    acc_val_curve = history.history["val_accuracy"]

    #Print them on 2 graphs
    plt.subplot(2,1,1)
    plt.plot(loss_curve, label="Training")
    plt.plot(loss_val_curve, label="Validation")
    plt.legend(frameon=False, loc='upper center', ncol=2)
    plt.xlabel('epochs')
    plt.ylabel('MODEL LOSS')
    plt.subplot(2,1,2)
    plt.plot(acc_curve, label="Training")
    plt.plot(acc_val_curve, label="Validation")
    plt.legend(frameon=False, loc='lower center', ncol=2)
    plt.xlabel('epochs')
    plt.ylabel('MODEL ACCURACY')
    plt.show()

def create_data_sets(training_folder, validation_folder, grey = True, two_classes = False) : 
    if two_classes :
        (trainX1, trainY1) = import_data(training_folder, CLASSES1, max_training, im_size = im_size, grey = grey)
        (valX1, valY1) = import_data(validation_folder, CLASSES1, max_validation, im_size = im_size, grey = grey)
        (trainX2, trainY2) = import_data(training_folder, CLASSES2, max_training, im_size = im_size, grey = grey)
        (valX2, valY2) = import_data(validation_folder, CLASSES2, max_validation, im_size = im_size, grey = grey)  
        trainY1.fill(0)
        valY1.fill(0)
        trainY2.fill(1)
        valY2.fill(1)
        trainX = np.concatenate((trainX1, trainX2))
        trainY = np.concatenate((trainY1, trainY2))
        valX = np.concatenate((valX1, valX2))
        valY = np.concatenate((valY1, valY2))
    else :
        (trainX, trainY) = import_data(training_folder, CLASSES, max_training, im_size = im_size, grey = grey)
        (valX, valY) = import_data(validation_folder, CLASSES, max_validation, im_size = im_size, grey = grey)
    # Shuffle the data
    # trainX, trainY = shuffle(trainX, trainY, random_state=0)
    # valX, valY = shuffle(valX, valY, random_state=0)
    # trainX, trainY = shuffle_in_unison(trainX, trainY)
    # valX, valY = shuffle_in_unison(valX, valY)
    # Reshaping training
    trainX = trainX.reshape(trainX.shape[0], im_size[0], im_size[1], nb_colors)
    valX = valX.reshape(valX.shape[0], im_size[0], im_size[1], nb_colors)
    return trainX, trainY, valX, valY

def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b


# =============================================================================
#     Constants
# =============================================================================

CLASSES1 = ['backpack', 'suitcase', 'handbag']
CLASSES2 = ['chair', 'bench', 'train', 'person']
CLASSES = CLASSES1 + CLASSES2
im_size = (64, 64, 1)
nb_colors = 1
NB_CLASSES = len(CLASSES)
max_training = 1000
max_validation = 100
# training_folder = "data_subsets/model_data_grey/training_set"
# validation_folder = "data_subsets/model_data_grey/validation_set"
training_folder = "CNN/data_subsets/model_data_grey3/training_set/"
                    #  CNN/data_subsets/model_data_grey3
validation_folder = "CNN/data_subsets/model_data_grey3/validation_set/"
grey = False
two_classes = False

if grey == False :
    im_size = (64, 64, 3)
    nb_colors = 3
    training_folder = "data_subsets/model_data_rgb2/training_set"
    validation_folder = "data_subsets/model_data_rgb2/validation_set"
if two_classes == True :
    NB_CLASSES = 2


# =============================================================================
#     Main
# =============================================================================

if __name__ == "__main__": 
    # Create training and validation sets
    trainX, trainY, valX, valY = create_data_sets(training_folder, validation_folder, grey, two_classes)
    print("[INFO] : Training and validation sets created")

    # Show a sample of images
    CLASSES = CLASSES1 + CLASSES2
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        if grey :
            plt.imshow(np.reshape(valX[i], (64, 64)), cmap="binary")
        else :
            plt.imshow(valX[i], interpolation='nearest')
        plt.xlabel(CLASSES[valY[i]])
    plt.show()

    # Create the model
    model = create_model(im_size)
    print("[INFO] : Model created")

    # Train the model
    (model, history) = Training(model, trainX, trainY, valX, valY)

    # Save the model
    model.save('cnn_models/simple_cnn.h5')

    # Plot curves
    plot_curves(history)

