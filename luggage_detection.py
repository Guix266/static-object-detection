# -*- coding: utf-8 -*-

"""
This module contains class for luggage detection.
All the methods are used to :
    - Dectect the static foreground
    - Classify the different static foreground objects and identify the luggages
    - Store the results
"""

import numpy as np
import pandas as pd
import os
import time
import cv2

import bg_models
from const import *
from utils import *

class LuggageDetection:

    def __init__(self, first_frame):
        ### shape have 3 channels
        self.image_shape = np.shape(first_frame)[0:2]
        shape_rgb = self.image_shape+(3,)
        
        ### Frame
        self.current_frame = first_frame.copy()

        ### Background substraction
        # define zivkovic background subtraction function
        self.f_bg_long = cv2.createBackgroundSubtractorMOG2(BG_ZIV_HIST, BG_ZIV_LONG_THRESH, True)
        self.f_bg_short = cv2.createBackgroundSubtractorMOG2(BG_ZIV_HIST, BG_ZIV_SHORT_THRESH, True)
        # define the corresponding foreground
        self.foreground_mask_long_term = np.zeros(shape=self.image_shape)
        self.foreground_mask_short_term = np.zeros(shape=self.image_shape)
        # Classic background
        # self.old_long_bg = np.zeros(shape=shape_rgb)
        # self.old_short_bg = np.zeros(shape=shape_rgb)

        ### Pixel states
        # define the matrix containing the state of each pixel : start as a background
        self.background_state= np.zeros(shape=self.image_shape, dtype=np.int8)
        # contain the mask that we finaly concider
        self.proposal_mask = np.zeros(shape=self.image_shape, dtype=np.uint8)  # mask from aggregator
        # Count the number of time the test is passed
        self.count = np.zeros(shape=self.image_shape, dtype=np.int8)  
                
        ### BOX
        # Contain the actual proposal boxes
        self.bbox = pd.DataFrame(columns = ['POS_X', 'POS_Y', 'WIDTH', 'HEIGHT', 'TIME_BEGIN', 'NB_FRAME', 'AGG_CLASS', 'LUGGAGE', 'ID_ALERT'])
        self.alert_nb = 0

    def compute_foreground_masks(self, frame):
        """
        Compute foreground masks for term background and short term background following Porikli's method
        Improve the result with postprocessing
        ==========================================================================
        :param np.uint8 frame: frame (3 channels) from which extract foregrounds masks
        
        :returns: foreground masks for long term and short term backgrounds
        :rtype: np.int8
        """

        # get rgb dual background (long and short sensitivity)
        # N.B. background is black (0) and foreground white (1)
        self.foreground_mask_long_term = bg_models.compute_foreground_mask_from_func(self.f_bg_long, frame,
                                                                                     BG_ZIV_LONG_LRATE)

        self.foreground_mask_short_term = bg_models.compute_foreground_mask_from_func(self.f_bg_short, frame,
                                                                                      BG_ZIV_SHORT_LRATE)
        
        ab = self.foreground_mask_short_term
        # self.foreground_mask_long_term, self.old_long_bg = bg_models.compute_foreground_classic(self.old_long_bg, frame,
        #                                                                              BG_ZIV_LONG_LRATE)
        # self.foreground_mask_short_term, self.old_short_bg = bg_models.compute_foreground_classic(self.old_short_bg, frame,
        #                                                                               BG_ZIV_SHORT_LRATE)
        
        # Opening
        self.foreground_mask_long_term = bg_models.apply_erosion(self.foreground_mask_long_term, 3, cv2.MORPH_ELLIPSE)
        self.foreground_mask_short_term = bg_models.apply_erosion(self.foreground_mask_short_term, 3, cv2.MORPH_ELLIPSE)
        
        # Dilation
        self.foreground_mask_long_term = bg_models.apply_dilation(self.foreground_mask_long_term, 4, cv2.MORPH_ELLIPSE)
        self.foreground_mask_short_term = bg_models.apply_dilation(self.foreground_mask_short_term, 4, cv2.MORPH_ELLIPSE)
        
        # Median filter
        self.foreground_mask_long_term = cv2.medianBlur(self.foreground_mask_long_term, 5)
        self.foreground_mask_short_term = cv2.medianBlur(self.foreground_mask_short_term, 5)
        
        ab2 = self.foreground_mask_short_term
        # Improvment of the forms recognition with watershed
        frame = cv2.blur(frame, (6,6))
        self.foreground_mask_long_term = bg_models.improved_foreground(frame, self.foreground_mask_long_term)
        self.foreground_mask_short_term = bg_models.improved_foreground(frame, self.foreground_mask_short_term)
        
        return ab, ab2, self.foreground_mask_long_term, self.foreground_mask_short_term


    def update_detection_state(self):
        """
        Update self.background_state with the provided foregrounds according to the FSM.
        Use a lookup table to optimize the performances 
        ==========================================================================
        :return bg_predic: Represent the values of self.foreground_mask_long_term and self.foreground_mask_short_term
                            in one single array
        :rtype: np.array
        """
        
        new_background_state = np.zeros(self.image_shape, dtype = np.uint8)
        
        # 00=0 / 10=1 / 01=2 / 11=3
        bg_predic = self.foreground_mask_long_term + 2*self.foreground_mask_short_term
        
        # Increment the states with the table
        for fg_type in range(4):
            for curent_state in states:
                bool_test = np.all(np.array([self.background_state==curent_state, bg_predic == fg_type]), axis=0)
                new_background_state[bool_test] = loockup_table[fg_type, curent_state]
        # print("min = {}".format(np.min(new_background_state)))
        
        # In case where a count down is needed : iterate self.count / reset the BG state to 0
        bool_test = np.any(np.array([self.background_state==CSF, self.background_state==OCSF]), axis=0)  # state CSF or OCSF
        self.count[bool_test] += 1
        self.count[self.background_state==BG] = -1
        
        # When the count reach FMI_BG_DIF and 
        bool_test = np.all(np.array([self.background_state==CSF, bg_predic == 1, self.count >= FMI_BG_DIF]), axis=0)   #goes in SFO
        new_background_state[bool_test] = SFO

        self.background_state =  new_background_state.copy()

        return(bg_predic)


    def extract_proposal_bbox(self, model):
        """
        Extract RGB proposal as the bounding boxes of the areas where the pixels are in the state SFO.
        Classify each bounding box and sort them into a DataFrame.
        ==========================================================================
        :param model: CNN model trained for classification with tensorflow. It has to have an input_size = (64,64,3)
                  The Output of the model is an vector with the probabilities related to each class from CLASSES constant
                  The first 3 classes have to be related to luggages and the other to backgrounds
        """

        ### create the proposal mask
        self.proposal_mask = np.where(self.background_state == SFO, 1, 0)
        self.proposal_mask = bg_models.apply_dilation(self.proposal_mask, 3, cv2.MORPH_ELLIPSE)
        self.proposal_mask = cv2.medianBlur(self.proposal_mask, 7)

        ### get a list with the new bounding boxes
        new_bbox = bg_models.get_bounding_boxes(self.proposal_mask.astype(np.uint8))

        ind = []    # Index of the boxes that will be deleted
        self.bbox.reset_index(drop=True, inplace=True)
        ### Process the bounding boxes and add them to the self.bbox
        for nbb in new_bbox:
            # Extend the box from 5 pixel on each side
            nb = extend_box(nbb, self.image_shape)
            # print("New_bow : ", nb, nbb)
            
            nb_frame = 0
            agg = 0
            lug = 0
            id_alert = 0
            time_b = time.asctime()
            # Determine to which old box the new one correspond
            for b in self.bbox.itertuples():
                iol = IoL(nb, (b.POS_X, b.POS_Y, b.WIDTH, b.HEIGHT))
                # print("\x1b[1;30;47miol = " + str(iol) +"\x1b[0m")
                if iol >= 0.5:
                    if b.LUGGAGE >= 1: #if luggage
                        iou = IoU(nb, (b.POS_X, b.POS_Y, b.WIDTH, b.HEIGHT))
                        if iou >= 0.7:  
                            if b.NB_FRAME >= nb_frame:
                                nb_frame = b.NB_FRAME
                                time_b = b.TIME_BEGIN
                                agg = b.AGG_CLASS
                                lug = b.LUGGAGE
                                id_alert = b.ID_ALERT
                            ind.append(b.Index)
                    else:   #If not lugguage
                        if b.NB_FRAME >= nb_frame:
                            nb_frame = b.NB_FRAME
                            time_b = b.TIME_BEGIN
                            agg = b.AGG_CLASS
                            lug = b.LUGGAGE
                            id_alert = b.ID_ALERT
                        ind.append(b.Index)
                
            # Add the new box on the DataFrame
            new_row = {'POS_X':nb[0], 'POS_Y':nb[1], 'WIDTH':nb[2], 'HEIGHT':nb[3], 'TIME_BEGIN':time_b, 
                       'NB_FRAME':nb_frame, 'AGG_CLASS':agg, 'LUGGAGE':lug, 'ID_ALERT':id_alert}
            self.bbox = self.bbox.append(new_row, ignore_index=True)
            
        # Remove what correspond to the old boxes
        self.bbox.drop(index = ind, inplace=True)
        # self.bbox.reset_index(drop=True, inplace=True)
        
        # Test with the CNN to iterate the agregator
        pre_lst = []
        for b in self.bbox.itertuples():
            pred_class, res = prediction_CNN(self.current_frame, (b.POS_X, b.POS_Y, b.WIDTH, b.HEIGHT), model)         
            if res >=CLASS_THRESH:  
                if b.AGG_CLASS < AGG_MAX:
                    pre_lst.append(b.AGG_CLASS + 1)
                else:
                    pre_lst.append(20)
            else:
                pre_lst.append(b.AGG_CLASS - 1)
            
            ### PRINT INFO NEW BOXES
            # for i in range(len(CLASSES)):
            #     if pred_class[i] >= 0.05:
            #         if i <3:
            #             message = '\x1b[5;30;42m' + CLASSES[i] +" = " + str(int(pred_class[i]*100)/100) + '\x1b[0m'
            #             print(message)
            #         else:
            #             message = '\x1b[6;30;41m' + CLASSES[i] +" = " + str(int(pred_class[i]*100)/100) + '\x1b[0m'
            #             print(message)
            
            # if res >= CLASS_THRESH:
            #     print('\x1b[5;30;42m Probas = '+str(int(res*100)/100)+'\x1b[0m')
            # else:
            #     print('\x1b[6;30;41m Probas = '+str(int(res*100)/100)+'\x1b[0m')
                
            # print("agg =", b.AGG_CLASS, ", pos = ", (b.POS_X, b.POS_Y, b.WIDTH, b.HEIGHT), "nb_frame =", b.NB_FRAME, "lugg = ", b.LUGGAGE )
        self.bbox.loc[:,'AGG_CLASS'] = pre_lst
                
        # Increment the boxes frame number 
        self.bbox.loc[:,'NB_FRAME'] += 1
        
        # Update the LUGGAGE value to 1 when a luggage is detected
        # (0=nothing, 1:Abandoned_luggage)
        self.bbox.loc[self.bbox['AGG_CLASS']==AGG_THRESH + 1, 'LUGGAGE'] = 1

        


    def process_proposal_bbox(self, database_path, folder_path):
        """
        Process the proposal bbox obtained :
            - Reset the pixel of non-luggage- objects
            - Add the needed bbox to the database and a copy of the image in a folder
            - When a bbox desapears, add the end date to the database
        ==========================================================================
        :param database_path: string containing the path of the database
        :param  folder_path: string containing the path of the folder where to save the alert images
        
        :return bbox: DataFrame of containing the infos about the bounding boxes
        :rtype: pd.DataFrame
        """
        # Reset the pixels of the non luggage proposal bbox on the class_RGB image
        self.background_state, self.bbox = bg_models.reset_bbox(self.background_state, self.bbox)
        
        # # Update the coordinates in the database if needed
        # df = self.bbox.loc[self.bbox['LUGGAGE']==1,:]
        # for b in df.itertuples():
        #     update_bd(database_path, b.ID_ALERT, b.POS_X, b.POS_Y, b.WIDTH, b.HEIGHT)
        
        
        # Add the needed bbox to the database and a copy of the image in a folder
        df = self.bbox.loc[(self.bbox['LUGGAGE']==0) & (self.bbox['AGG_CLASS']==AGG_THRESH),:]
        ids = []
        for b in df.itertuples():
            # Copy the image in a folder
            image = draw_bounding_box(self.current_frame, (b.POS_X, b.POS_Y, b.WIDTH, b.HEIGHT))
            image_path = folder_path +"/alert_image_"+str(self.alert_nb)+".jpg"
            cv2.imwrite(image_path, image)
            self.alert_nb += 1
            
            # Add to db
            id_alert = write_bd(database_path, image_path, b.POS_X, b.POS_Y, b.WIDTH, b.HEIGHT, b.TIME_BEGIN)
            ids.append(id_alert)
        self.bbox.loc[(self.bbox['LUGGAGE']==0) & (self.bbox['AGG_CLASS']==AGG_THRESH),"ID_ALERT"] = ids
        
        # Write the ending time for those that stopped to be detected
        df = self.bbox.loc[(self.bbox['LUGGAGE']==1) & (self.bbox['AGG_CLASS']==AGG_THRESH),:]
        for b in df.itertuples():
            time_end_bd(database_path, b.ID_ALERT)
        
        return  self.bbox
