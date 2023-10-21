# -*- coding: utf-8 -*-

"""
This module contains different functions used in the program.
"""

import numpy as np
import time
import cv2
import sqlite3
from tensorflow.keras.applications.resnet50 import preprocess_input

from const import *

# =============================================================================
#  BOX MANAGEMENT
# =============================================================================

def get_center_area_from_rect(rect):
    #print "rect: ", rect
    """ coordinates rect center """
    cx = rect[0] + rect[2] / 2
    cy = rect[1] + rect[3] / 2
    area = area = rect[2] * rect[3]
    return cx, cy, area

def norm_correlate(a, v):
    a = (a - np.mean(a)) / (np.std(a) * len(a))
    v = (v - np.mean(v)) / np.std(v)

    return np.correlate(a, v)

# (s[0], s[1]), (s[0]+s[2], s[1]+s[3])
def boxes_intersect(bbox1, bbox2):
    """ Return if two rect overlap """
    return ((np.abs(bbox1[0]-bbox2[0])*2) < (bbox1[2]+bbox2[2])) and ((np.abs(bbox1[1]-bbox2[1])*2) < (bbox1[3]+bbox2[3]))


def boxes_intersect2(bbox1, bbox2):
    """ Return if two rect overlap """
    def in_range(value, min, max):
        return (value >= min) and (value <= max)

    x_overlap = in_range(bbox1[0], bbox2[0], bbox2[0]+bbox2[2]) or in_range(bbox2[0], bbox1[0], bbox1[0]+bbox1[2])
    y_overlap = in_range(bbox1[1], bbox2[1], bbox2[1]+bbox2[3]) or in_range(bbox2[1], bbox1[1], bbox1[1]+bbox1[3])

    return x_overlap and y_overlap

    
def IoU(boxA, boxB):
    """ Return the intersection over Union of the two bounding boxes """
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
    yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    
    # Area of rectangles
    boxAArea = (boxA[2]+1) * (boxA[3]+1)
    boxBArea = (boxB[2]+1) * (boxB[3]+1)
    
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return(iou)

def IoL(boxA, boxB):
    """ 
    Return the Area of the intersection over the area of the smallest box
    Represent how much a box is inside an other 
    Input bbox: ( X, Y, W, H) """
        
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
    yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    
    # Area of rectangles
    boxAArea = (boxA[2]+1) * (boxA[3]+1)
    boxBArea = (boxB[2]+1) * (boxB[3]+1)
    
    iol = interArea / min(boxAArea, boxBArea)
    return(iol)
    
def extend_box(box, imshape):
    """Extend the box from 5 pixel on each side"""
    add = 5
    height, width = imshape
    r1 = min( box[0], add)
    r2 = min( box[1], add)
    r3 = min( width - box[0] - 1, add + box[2])
    r4 = min( height - box[1] - 1, add + box[3])
    nbox = (box[0]-r1, box[1]-r2, r3 +add, r4 + add)
    return(nbox)
    
# print(extend_box((10,20,20,40),(33, 62)))  

# =============================================================================
#  Manage time
# =============================================================================

def time_difference(time1, time2):
    """ Return the difference between 2 times """
    t1 = time.mktime(time.strptime(time1))
    t2 = time.mktime(time.strptime(time2))
    
    seconds = abs(t1-t2)
    
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    if hour == 0:
        return "%02d:%02d" % (minutes, seconds) 
    else:
        return "%d:%02d:%02d" % (hour, minutes, seconds) 
    
# =============================================================================
#  Draw bbox    
# =============================================================================

def draw_bounding_box(image, bbox):
    """ Draw all bounding box inside image as red rectangle
    ==========================================================================
    :param image: image where draw the bounding box
    :param bbox: list of bounding boxes as (x,y,w,h) where x,y is the topleft corner of the rectangle
    
    :return: image with bbox drawn
    :rtype: np.array
    """
    image2 = image.copy()
    cv2.rectangle(image2, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0,0,255), 3)
    cv2.putText(image2, "LUGGAGE", (bbox[0],bbox[1]-10),1,1.5,(0,0,255))
    
    t = time.asctime()
    height, width = np.shape(image)[:2]

    text_org = (int(0.05*width), int(0.95*height))
    cv2.putText(image2, t, text_org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), thickness=2)
    return image2


def draw_bounding_box_df(image, bbox, agg_lim):
    """ 
    Draw all bounding box inside image as red rectangle from a dataframe
    ==========================================================================
    :param image: image where draw the bounding box
    :param bbox: DataFrame of containing the infos about the bounding boxes
    
    :return: image with bbox drawn
    :rtype: np.array
    """
    for s in bbox.itertuples():
        if s.AGG_CLASS >= agg_lim:
            # Time calculation
            t = time.asctime()
            t = time_difference(t, s.TIME_BEGIN)
            
            cv2.rectangle(image, (s.POS_X, s.POS_Y), (s.POS_X+s.WIDTH, s.POS_Y+s.HEIGHT), (0,0,255), 3)
            cv2.putText(image, "SUITCASE t = {}".format(t), (s.POS_X,s.POS_Y-10),1,1.5,(0,0,255))
        else:
            cv2.rectangle(image, (s.POS_X, s.POS_Y), (s.POS_X+s.WIDTH, s.POS_Y+s.HEIGHT), (255,0,0), 2)
            cv2.putText(image, "UNDETERMINED", (s.POS_X,s.POS_Y-10),1,1.5,(255,0,0))
    return image

# =============================================================================
#  Give the probability for the bbpx to be a luggage
# =============================================================================

def prediction_CNN(image, bbox, model):
    """ 
    Give the probability for the bbpx to be a luggage thanks to a CNN model
    ==========================================================================
    :param image: original RGB image
    :param bbox: DataFrame of bounding boxes as (x,y,w,h) where x,y is the topleft corner of the rectangle
    :param model: CNN model trained with tensorflow. It has to have an input_size = (64,64,3)
                  The Output of the model is an vector with the probabilities related to each class from CLASSES constant
    
    :return Ypred: Vector containing the probabilities related to each class from CLASSES
    :rtype: np.array
    :return Ypred: Probability for the box to be a luggage (suitcase or handback or backpack)
    :rtype: float
    """
    crop_im = image[bbox[1]:bbox[1]+bbox[3], bbox[0]: bbox[0]+bbox[2]]

    image = cv2.resize(crop_im, (64,64), interpolation = cv2.INTER_AREA)
    image = preprocess_input(image)
    image = image.reshape(1, 64, 64, 3)
    Ypred = np.array(model.predict(image)[0])
    
    res = sum(Ypred[:3])
    return(Ypred, res)

# =============================================================================
#  Data Base Management
# =============================================================================

def write_bd(database_path, image_path, pos_x, pos_y, width, height, time_begin):
    """ 
    Write a box in a row into the database
    ==========================================================================
    :param database_path: string containing the path of the database
    :param image_path: string containing the path of the alert image
    :param (pos_x, pos_y, width, height): bbox values
    :param time_begin: String containing the time when the luggage appears 
                    (ex:'Tue Jul 28 17:39:31 2020')        
                    
    :return id_alert: ID of the row written in the database
    :rtype: int
    """
    try:
        connection = sqlite3.connect(database_path)
        cursor = connection.cursor()
                # ID_ALERT    POS_X    POS_Y    WIDTH  HEIGHT TIME_BEGIN  TIME_END  TIME_TOTAL  IMAGE_PATH  
        alert = (cursor.lastrowid, pos_x, pos_y, width, height, time_begin, None, None, image_path)
        req = cursor.execute("INSERT INTO bagage_alerte VALUES(?,?,?,?,?,?,?,?,?)", alert)
        connection.commit()
        
        req = cursor.execute("SELECT MAX(ID_ALERT) FROM bagage_alerte")
        id_alert = int(cursor.fetchone()[0])
        
    except Exception as e:
        print("[ERREUR] :", e)
        connection.rollback()
    finally:
        connection.close()
        return(id_alert)

# write_bd("Output/results.db", 1, 2, 3, 4, 2)

def update_bd(database_path, id_alert, pos_x, pos_y, width, height):
    """
    Update the position of the bags in the db
    ==========================================================================
    :param database_path: string containing the path of the database
    :param id_alert: ID of the row written in the database
    :param (pos_x, pos_y, width, height): bbox values     
    """
    try:
        connection = sqlite3.connect(database_path)
        cursor = connection.cursor()
        req = cursor.execute("UPDATE bagage_alerte SET POS_X=?, POS_Y=?, WIDTH=?, HEIGHT=? WHERE ID_ALERT=?", 
                             (pos_x, pos_y, width, height, id_alert))
        connection.commit()
    except Exception as e:
        print("[ERREUR] :", e)
        connection.rollback()
    finally:
        connection.close()
        
  
def time_end_bd(database_path, id_alert):
    """ 
    Write the final time in the db when the bag disappear
    ==========================================================================
    :param database_path: string containing the path of the database
    :param id_alert: ID of the row written in the database  
    """
    try:
        connection = sqlite3.connect(database_path)
        cursor = connection.cursor()
        
        req = cursor.execute("SELECT TIME_BEGIN FROM bagage_alerte WHERE ID_ALERT = ?", (str(id_alert),)) 
        time_begin = cursor.fetchone()[0]
        time_end = time.asctime()
        time_dif = time_difference(time_begin, time_end)
        
        req = cursor.execute("UPDATE bagage_alerte SET TIME_END=?, TIME_TOTAL=?  WHERE ID_ALERT=?", 
                              (time_end, time_dif, id_alert))
        connection.commit()
    except Exception as e:
        print("[ERREUR] :", e)
        connection.rollback()
    finally:
        connection.close()

# time_end_bd("Output/results.db", 32)

def print_db(database_path):
    """
    Print the elements of the database
    ==========================================================================
    :param database_path: string containing the path of the database
    """
    connection = sqlite3.connect(database_path)
    cursor = connection.cursor()
    req = cursor.execute("SELECT * FROM bagage_alerte")
    id_alert = cursor.fetchall()
    print(id_alert)
    connection.close()
    
def reset_db(database_path):
    """
    Reset the values of the database
    ==========================================================================
    :param database_path: string containing the path of the database
    """
    try:
        connection = sqlite3.connect(database_path)
        cursor = connection.cursor()
        req = cursor.execute("DELETE FROM bagage_alerte")
        connection.commit()
    except Exception as e:
        print("[ERREUR] :", e)
        connection.rollback()
    finally:
        connection.close()


    
    