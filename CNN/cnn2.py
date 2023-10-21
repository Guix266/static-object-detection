# -*- coding: utf-8 -*-

"""
Module used to train a CNN model
"""

import tensorflow as tf

# import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
from sklearn.preprocessing import StandardScaler

import random as r
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os


def import_data(folder_name, classes, n, im_size = (64, 64)):
    size = n*len(classes)
    X = np.zeros((size, im_size[0], im_size[1]), dtype = np.uint8)
    Y = np.zeros(size, dtype = np.uint8)
    j = 0
    for category in classes:
        print("\r[INFO] : Import of class "+category, end='')
        for i in range(n):
            num = str(i)
            while len(num) < 5:
                num = "0"+num
            im = os.path.join(folder_name +"\\" + category, category + "_" + num +".jpg")
            X[j*n + i,:,:] = np.array(Image.open(im))
            Y[j*n + i] = j
        j += 1
    return(X, Y)

def import_all_data(folder_name, classes, im_size = (64, 64, 1), biclass = False):
    """ Import all the datat from the folders """
    class_size = []
    for category in classes:    
        class_size.append(len(os.listdir(os.path.join(folder_name, category))) -1)
    size = sum(class_size)

    X = np.zeros((size, im_size[0], im_size[1], im_size[2]), dtype = np.uint8)
    Y = np.zeros(size, dtype = np.uint8)
    k = 0
    
    if biclass == False:
        for j in range(len(classes)):
            print("[INFO] : Import class "+str(j)+"/"+str(len(classes)))
            for i in range(class_size[j]):
                num = str(i)
                while len(num) < 5:
                    num = "0"+num
                im = os.path.join(folder_name +"\\" + classes[j], classes[j] + "_" + num +".jpg")
                img = np.array(Image.open(im))
                if len(np.shape(img)) == 2:
                    img2 = np.zeros(im_size)
                    img2[:,:,r.randrange(3)] = img
                    img = img2
                X[k + i,:,:,:] = img
                Y[k + i] = j
            k += class_size[j]
    elif biclass == True:
        for j in range(len(classes)):
            print("[INFO] : Import class "+classes[j])
            for i in range(class_size[j]):
                num = str(i)
                while len(num) < 5:
                    num = "0"+num
                im = os.path.join(folder_name +"\\" + classes[j], classes[j] + "_" + num +".jpg")
                img = np.array(Image.open(im))
                if len(np.shape(img)) == 2:
                    img2 = np.zeros(im_size)
                    img2[:,:,r.randrange(3)] = img
                    img = img2
                X[k + i,:,:,:] = img
                if j <= 2:
                    Y[k + i] = 1
                else:
                    Y[k + i] = 0
            k += class_size[j]        
    return(X, Y)


def create_model(im_size):
    """creation of the CNN model with 2 convolutions layers"""
    
    model = Sequential([
        Conv2D(64, (3,3), padding='same', activation='relu',input_shape=im_size),
        # Conv2D(64, (3, 3), padding='same', activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.2),
        Conv2D(32, (3,3), padding='same', activation='relu'),
        # Conv2D(32, (3, 3),padding='same', activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(input_shape=im_size),
        Dense(128, activation='relu'),
        Dense(NB_CLASSES, activation='softmax')
    ])
    
    #Compilation of the model
    model.compile(  loss = 'categorical_crossentropy',
                    optimizer ='Adam',
                    metrics = ['accuracy'])
    return(model)


def Training(model, trainX, trainY, testX, testY, checkpoint_path):
    """Train the input model with 25 epochs"""
    print("[INFO] : Training of the CNN ...")
    
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     monitor='val_accuracy', 
                                                     save_best_only=True,
                                                     verbose=1)

    earlystop_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', min_delta=0.0001,
        patience=5)
    
    history = model.fit(trainX, trainY,
                        batch_size= 100,
                        epochs=7,
                        validation_data=(testX, testY),
                        verbose = 1,
                        callbacks=[earlystop_callback, cp_callback])
    return(model, history)


def plot_curves(history):
    """Trace learning curve/accuracy rate"""
    # Get the values
    loss_curve = history.history["loss"]
    acc_curve = history.history["accuracy"]
    loss_val_curve = history.history["val_loss"]
    acc_val_curve = history.history["val_accuracy"]

    #Print them on 2 graphs
    fig, (ax1, ax2) = plt.subplots(2, 1)

    ax1.plot(loss_curve, label="Training")
    ax1.plot(loss_val_curve, label="Validation")
    ax1.legend(frameon=False, loc='upper center', ncol=2)
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('MODEL LOSS')
    
    ax2.plot(acc_curve, label="Training")
    ax2.plot(acc_val_curve, label="Validation")
    ax2.legend(frameon=False, loc='lower center', ncol=2)
    ax2.set_xlabel('epochs')
    ax2.set_ylabel('MODEL ACCURACY')
    plt.show()
    return(fig)

def processed_data(biclass):
    # =============================================================================
    #       TRAINING
    # =============================================================================
    
    ### Import the classes :
    # (X, Y) = import_data("CNN\data_subsets\model_data", CLASSES, 6000, im_size = im_size)
    (X, Y) = import_all_data("data_subsets\model_data", CLASSES, im_size = im_size, biclass = biclass)
    # Data analysis
    if analysis == True:
        unique_elements, counts_elements = np.unique(Y, return_counts=True)
        x = unique_elements
        plt.bar(x, counts_elements)
        plt.xlabel('Class')
        plt.ylabel('Frequency')
        plt.title("Original dataset distribution")
        plt.show()
    
    ### Dataset preprocessing :
    # Create Hot-Encoder for the labels
    Y = tf.keras.utils.to_categorical(Y, NB_CLASSES, np.int32)
    
    # Normalize the data
    # X = X / 255
    
    # Standardize the data
    # datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    #     featurewise_center=True, featurewise_std_normalization=True)
    # datagen.fit(trainX)
    
    # 80% training set and 20% test set and normalization
    trainX, testX, trainY, testY = train_test_split(X,Y, test_size=0.2, random_state=0)
    # Reshaping the trainset for training
    trainX = trainX.reshape(trainX.shape[0], im_size[0], im_size[1], 1)
    testX = testX.reshape(testX.shape[0], im_size[0], im_size[1], 1)
    
    return(trainX, testX, trainY, testY)

# =============================================================================
#     MODEL  PARAMETERS
# =============================================================================

checkpoint_path = "models/checkpoint/cp.ckpt"

CLASSES = ['backpack', 'suitcase', 'handbag', 'bench', 'chair', 'person', 'train']
NB_CLASSES = len(CLASSES)
im_size = (64, 64, 1)
biclass = False

analysis = False



if __name__ == "__main__":
    
    # Import the data processed
    (trainX, testX, trainY, testY) = processed_data(biclass)
    
    ### Create the model
    model = create_model(im_size)
    
    ### Train the model
    (model, history) = Training( model, trainX, trainY, testX, testY, checkpoint_path)

    # model.save('models/simple_cnn_2class.h5')
    
    plot_curves(history)
