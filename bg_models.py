# -*- coding: utf-8 -*-

"""
This module contains the functions for image processing and foreground detection
"""

import numpy as np
import cv2

from const import *

def improved_foreground(image, mask):
    """ 
    Improve the binarization of the image with a watershed calculus 
    The detected forms are more accurates with this process
    ==========================================================================
    :param image: image RGB from the camera (self.current_frame)
    :param mask: foreground masks for long term or short term backgrounds (self.foreground_mask_long_term)
    
    :return: new foreground masks
    :rtype: np.uint8
    """
    
    # get sure background area    
    kernel = np.ones((5,5),np.uint8)
    sure_bg = cv2.dilate(mask, kernel, iterations = 5)
    sure_fg = cv2.erode(mask, kernel, iterations = 2)
    unknown = cv2.subtract(sure_bg,sure_fg)
    
    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    # Now, mark the region of unknown with zero
    markers[unknown==1] = 0
    # label_preshed = cv2.applyColorMap(np.uint8(markers)*20, cv2.COLORMAP_HSV)
    
    # watershed
    markers = cv2.watershed(image, markers)
    # label_postshed = cv2.applyColorMap(np.uint8(markers)*20, cv2.COLORMAP_HSV)
    new_bg = np.where(markers == 1, 0, 1)
    return(new_bg.astype(np.uint8))


def compute_foreground_mask_from_func(f_bg, current_frame, alpha):
    """
    Extract binary foreground mask (1 foreground, 0 background) from f_bg background modeling function and update
    background model.
    ==========================================================================
    :param f_bg: background modeling function
    :param current_frame: current frame from which extract foreground
    :param alpha: update learning rate
    
    :return: foreground mask
    :rtype: np.uint8
    """
    foreground = np.zeros(shape=current_frame.shape, dtype=np.uint8)
    
    # We get the foreground and background : 
    # Pixels values : BG = 0, FG = 255, Shadows = 127
    foreground = f_bg.apply(current_frame, foreground, alpha)
   
    # convert to 0 1 notation to only keep the foreground : 255 => 1 , 0 => 0, 127 => 0
    foreground = np.where((foreground == 255), 1, 0)
    return foreground

def compute_foreground_classic(old_bg, current_frame, alpha) :
    """ Compute the simple background substraction of 2 frames to update the background detection
    NOT USED
    """
    foreground1 = cv2.subtract(old_bg, current_frame)
    foreground2 = cv2.subtract(current_frame, old_bg)
    foreground = cv2.add(foreground1, foreground2)
    foreground = cv2.cvtColor(foreground, cv2.COLOR_BGR2GRAY)
    foreground = np.where(foreground >= 40, 1, 0)
    new_bg = cv2.addWeighted(old_bg, (1-alpha), current_frame, alpha, 0)
    return foreground, new_bg


def apply_erosion(image, kernel_size, kernel_type):
    """
    Apply opening to image with the specified kernel type and image
    ==========================================================================
    :param image:   image to which apply opening
    :param kernel_size: size of the structuring element
    :param kernel_type: structuring element
    
    :return: image with opening applied
    :rtype: np.uint8
    """
    u_image = image.astype(np.uint8)
    #foreground_mask_depth = foreground_mask_depth.astype(np.uint8)
    kernel = cv2.getStructuringElement(kernel_type, (kernel_size, kernel_size))
    u_image = cv2.morphologyEx(u_image, cv2.MORPH_ERODE, kernel)
    u_image = cv2.morphologyEx(u_image, cv2.MORPH_ERODE, kernel)
    return u_image

def apply_dilation(image, kernel_size, kernel_type):
    """
    Apply dilation to image with the specified kernel type and image
    ==========================================================================
    :param image:   image to which apply opening
    :param kernel_size: size of the structuring element
    :param kernel_type: structuring element
    
    :return: image with opening applied
    :rtype: np.uint8
    """
    u_image = image.astype(np.uint8)
    #foreground_mask_depth = foreground_mask_depth.astype(np.uint8)
    kernel = cv2.getStructuringElement(kernel_type, (kernel_size, kernel_size))
    u_image = cv2.morphologyEx(u_image, cv2.MORPH_DILATE, kernel)
    u_image = cv2.morphologyEx(u_image, cv2.MORPH_DILATE, kernel)
    return u_image

def apply_opening(image, kernel_size, kernel_type):
    """ opening morphology"""
    u_image = image.astype(np.uint8)
    # Erode
    u_image = apply_erosion(image, kernel_size, kernel_type)
    # Dilate
    u_image = apply_dilation(image, kernel_size, kernel_type)
    return u_image

def get_bounding_boxes(image):
    """
    Return Bounding Boxes in the format x,y,w,h where (x,y) is the top left corner
    ==========================================================================
    :param image: image from which retrieve the bounding boxes
    
    :return: bounding boxes list
    :rtype: list
    """
    bbox = []
    contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        # filter contours with area less than 50 pixel
        if cv2.contourArea(cnt) > BBOX_MIN_AREA:
            rect = cv2.boundingRect(cnt)
            if rect not in bbox:
                bbox.append(rect)

    return bbox


def get_bounding_boxes2(image):
    """
    Return Bounding Boxes in the format x,y,w,h where (x,y) is the top left corner
    ==========================================================================
    :param image: image from which retrieve the bounding boxes
    
    :return: bounding boxes array, where each element has the form (x, y, w, h, counter) with counter = 1
    :rtype: np.array
    """
    squares = []
    bbox_elements = np.array([], dtype=int)
    contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        # filter contours with area less than 50 pixel
        if cv2.contourArea(cnt) > BBOX_MIN_AREA:
            rect = cv2.boundingRect(cnt)
            if rect not in squares:
                squares.append(rect)
                if bbox_elements.size == 0:
                    # save bbox with a counter set to one
                    bbox_elements = np.array([[rect[0], rect[1], rect[2], rect[3], 1]])
                else:
                    bbox_elements = np.concatenate((bbox_elements, [[rect[0], rect[1], rect[2], rect[3], 1]]))
    return bbox_elements


def reset_bbox(image, bbox_lst):
    """
    Reset the pixel inside of non luggage boxes
    It happens when AGG_CLASS became inferior to a minimum constant AGG_MIN
    Suppress the box from the bbox_list
    ==========================================================================
    :param image: image from which reset the pixel (SELF.background_state)
    :param bbox_lst: dataframe of bounding boxes as (x,y,w,h) where x,y is the topleft corner of the rectangle
    
    :return image: New image
    :rtype: np.array
    :return bbox_lst: new bounding boxes dataframe
    :rtype: pd.DataFrame
    
    """
    for box in bbox_lst.loc[bbox_lst.AGG_CLASS<= AGG_MIN].itertuples():
        for i in range(box.POS_X, box.POS_X + box.WIDTH):
                for j in range(box.POS_Y, box.POS_Y + box.HEIGHT) :
                    image[j,i] = BG
        
    # Remove the boxes that are not bags
    bbox_lst = bbox_lst.loc[bbox_lst.AGG_CLASS> -15]
    
    return(image, bbox_lst)

