# -*- coding: utf-8 -*-

"""
Module used to test a CNN model
"""

import tensorflow as tf

# import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
from sklearn.preprocessing import StandardScaler
import sklearn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import os
import sys
import seaborn as sn
import cv2

import cnn2


from tensorflow.keras.applications.resnet50 import ResNet50, decode_predictions, preprocess_input

def accuracy_luggage(Ypred, testY):
    """ Ypred : Hot encoder of the prediction
        testY : Ground trust"""
    n, m = np.shape(Ypred)
    Yp = np.zeros(n)
    
    # 1 = sac (class 'backpack', 'suitcase', 'handbag') 0 = reste
    Y_true = np.sum(testY[:,:3], axis = 1)
 
    for i in range(n):
        M = Ypred[i,0]
        ind = 0
        for j in range(1,m):
            if Ypred[i,j] >= M:
                M = Ypred[i,j]
                ind = j
        # Is the M in the 3 first class ?
        if ind <= 2:
            Yp[i] = 1
    
    Equ = (Yp == Y_true)
    unique_elements, counts_elements = np.unique(Equ, return_counts=True)
    Result_false = np.zeros((counts_elements[0], m))
    Result_true = np.zeros((counts_elements[1], m))
    j = 0
    for i in range(n):
        if Equ[i] == True:
            Result_true[j,:] = Ypred[i,:]
            j += 1
        else:
            Result_false[i-j,:] = Ypred[i,:]
            
    
    acc = counts_elements[1] / n
    
    return(Yp, Y_true, acc, Result_false, Result_true)
      

def import_FSM_result(folder_name, im_size):
    list_folder = ['other', 'valise']
    class_size = []
    for category in list_folder:    
        class_size.append(len(os.listdir(os.path.join(folder_name, category))))
    size = sum(class_size)
    
    X = np.zeros((size, im_size[0], im_size[1], im_size[2]), dtype = np.uint8)
    df = pd.DataFrame(columns=['image_name', 'test_result', 'ground_trust'])

    k = 0
    l = 0
    for i in range(len(list_folder)):
        print("[INFO] : Import images "+list_folder[i])
        path = os.path.join(folder_name, list_folder[i])
        for j in range(class_size[i]):
            name = list_folder[i]+"_"+str(j)+".jpg"
            im = Image.open(os.path.join(path, name))
            # im = im.convert('L')                                #Greyscale
            im = im.resize((im_size[0], im_size[1]), resample=Image.BILINEAR)     # Resize
            X[k,:,:] = np.array(im)
            df.loc[k, 'image_name'] = name
            df.loc[k, 'ground_trust'] = l
            # im.save(os.path.join(path, image))
            k += 1
        l += 1
    return(X, df)
  

def max_result(Ypred, bi_class = True):
    n, m = np.shape(Ypred)
    Yp = np.zeros(n, dtype=np.uint8)

    for i in range(n):
        M = Ypred[i,0]
        ind = 0
        for j in range(1,m):
            if Ypred[i,j] >= M:
                M = Ypred[i,j]
                ind = j
        if bi_class == True:
            # Is the M in the 3 first class ?
            if ind <= 2:
                # 1 = sac (class 'backpack', 'suitcase', 'handbag') 0 = reste
                Yp[i] = 1
        else:
            Yp[i] = ind
    return(Yp)


def conf_mat(Y_true, Yp):
    mat = confusion_matrix(Y_true, Yp)
    prob_mat = np.zeros(np.shape(mat))
    elements = np.sum(mat, axis=1)
    for i in range(np.shape(mat)[0]):
        prob_mat[i,:] = mat[i,:]/elements[i]
    
    df_cm = pd.DataFrame(prob_mat)
    # plt.figure(figsize=(10,7))
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})#, fmt='g') # font size
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    
    
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

def repartition(Y):
    unique_elements, counts_elements = np.unique(Y, return_counts=True)
    x = unique_elements
    plt.bar(x, counts_elements)
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.title("Original dataset distribution")
    plt.show()
    
# =============================================================================
#       Charge an existing model / data
# =============================================================================

# model.load_weights(checkpoint_path)
model = tf.keras.models.load_model('models/pretrain_cnn.h5')

# =============================================================================
#       Test sur le validation set
# =============================================================================
#%%

CLASSES = ['backpack', 'suitcase', 'handbag', 'bench', 'chair', 'person', 
          'train', 'bicycle', 'motorcycle', 'bus', 'truck', 'car', 'fire hydrant' ]
NB_CLASSES = len(CLASSES)
im_size = (64, 64, 3)

(Xtrain, Ytrain) = cnn2.import_all_data("data_subsets\\train2017", CLASSES, im_size = im_size)
(Xval, Yval) = cnn2.import_all_data("data_subsets\\val2017", CLASSES, im_size = im_size)

# randomize
X = np.concatenate((Xtrain, Xval), axis=0)
y = np.concatenate((Ytrain, Yval), axis=0)
X, y = sklearn.utils.shuffle(X, y)

# Downsampling
# X, y = downsampling(X, y)
repartition(y)
y = tf.keras.utils.to_categorical(y, NB_CLASSES, np.int32)
y.astype(np.int8)

# Preprocess the images
X = preprocess_input(X)


### I)  Test de la classification
# model.evaluate(testX, testY)

# Yp = max_result(Ypred, bi_class = False)
# Y_true = max_result(testY, bi_class = False)
# conf_mat(Y_true, Yp)



### II) Test du problème baggage
Ypred = model.predict(X)
thresh = 0.2

# Yp : prediction du model = bagage 1 ou non 0
# Ytrue : Ground trust = bagage 1 ou non 0
# acc : accuracy de la détection des 3 classes de bagages 
Y_true = np.sum(y[:10000,:3], axis=1)
Yp = (np.sum(np.array(Ypred)[:,:3],axis=1)>=thresh )*1
# Yp=max_result(Ypred, bi_class = True)

confusion_matrix(Y_true, Yp)

conf_mat(Y_true, Yp)

# =============================================================================
#       Test sur le résultat du background substraction
# =============================================================================
#%%

# # Import dataset
# Xtest, df = import_FSM_result("C:\\Users\\guill\\OneDrive\\Documents\\Stage_PIMAN\\_Code\\detection-d-objets-statiques\\Output\\images", (64, 64, 3))
# Y_true = np.array(df['ground_trust'])

# # Prédiction avec le model
# preprocess_Xtest = preprocess_input(Xtest)
# Y_FSM_pred = model.predict(preprocess_Xtest.reshape(np.shape(Xtest)[0], 64, 64, 3))
# # Y_FSM_pred = Y_FSM_pred[:,1]
# Y_FSM_pred = max_result(Y_FSM_pred)
# df["test_result"] = np.array(Y_FSM_pred.astype(np.int8))

# # Ecriture dans csv
# df.to_csv('C:\\Users\\guill\\OneDrive\\Documents\\Stage_PIMAN\\_Code\\detection-d-objets-statiques\\Output\\images\\model_predict.csv', index=False)

# # Matrice confusion
# conf_mat(Y_true.astype(np.int8), Y_FSM_pred.astype(np.int8))

#~~~~~~~~~~~~ CHECK IMAGES
#%%

# cv2.namedWindow('image',cv2.WINDOW_NORMAL)
# cv2.resizeWindow('image', 600,600)
# # PLot imported images
# for i in range(np.shape(Xtest)[0]):
#     image = Xtest[i,:,:]
#     cv2.imshow("image", image)
#     # print(df.loc[i, 'image_name'])
#     a = str(df.loc[i, 'ground_trust'])
#     b = str(df.loc[i, 'test_result'])
#     print(str(i)+" res : "+str(a==b)+"\t True: "+a+"\t prediction:"+b)
#     key = cv2.waitKey(0) & 0xFF
#     if key == ord("q"):
#         break

