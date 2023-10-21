# -*- coding: utf-8 -*-
"""
Main program to treat images from a url video stream.
"""

import numpy as np
import sys
import sqlite3
import tensorflow as tf
import time
from imutils.video import FPS

from luggage_detection import *
from const import *
import import_images

# =============================================================================
#     Import CNN
# =============================================================================

model = tf.keras.models.load_model('CNN/models/pretrain_cnn.h5')

# =============================================================================
# Connection BDD
# =============================================================================

database_path = "Output/results.db"
folder_path = "Output/alert_images"

conn = sqlite3.connect(database_path)
c = conn.cursor()
c.execute(''' CREATE TABLE IF NOT EXISTS"bagage_alerte" (
                	"ID_ALERT"	INTEGER,
                	"POS_X"	INTEGER,
                	"POS_Y"	INTEGER,
                	"HEIGHT"	INTEGER,
                	"WIDTH"	INTEGER,
                	"TIME_BEGIN"	TEXT,
                	"TIME_END"	TEXT,
                	"TIME_TOTAL"	TEXT,
                    "IMAGE_PATH"   TEXT,
                	PRIMARY KEY("ID_ALERT" AUTOINCREMENT))''')	
conn.commit()
conn.close()


# print_db(database_path)
# reset_db("database_path)


# =============================================================================
#    Image source : https://www.insecam.org/en/
# =============================================================================
#%%

# Supermarket
# link = "http://212.154.245.179/jpg/image.jpg?1592831139"
# Beach
# link = "http://97.68.104.34:80/mjpg/video.mjpg"
# Parking
# link = "http://185.138.104.204:8081/mjpg/video.mjpg"
# Road
# link = "http://217.170.99.226:8080/cgi-bin/viewer/video.jpg?r=1592902839"


# Camera URL from text
f = open("links.txt", "r")
link = f.readline()
link = f.readline()
link = link.rstrip('\n')
f.close()
 

cap = cv2.VideoCapture(link)
time.sleep(1.0)
fps = FPS().start()

fps_cam = round(cap.get(cv2.CAP_PROP_FPS))
print("[INFO]: {} Frames Per Second".format(fps_cam))

ret, frame = cap.read()
cap.release() 
# cv2.imshow('Video', frame)
# cv2.waitKey(1)

# cv2.namedWindow("Result",cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Result", 1800,800)

# Create instance
rgb = LuggageDetection(frame)
print("[INFO]: Detection ready")
print("Press 'q' to leave")


second = int(time.time() * ratio)

while(1):
    if second != int(time.time() * ratio) :
        second = int(time.time() * ratio)
        # print(second)

        # =============================================================================
        #  Import image    
        # =============================================================================
    
        # Import the image from the URL
        cap = cv2.VideoCapture(link)
        ret, rgb.current_frame = cap.read()
        cap.release() 
    
        # =============================================================================
        #  Algorithm
        # =============================================================================
        
        # get rgb dual background (long and short sensitivity)
        # N.B. background is black (0) and foreground white (1)
        rgb.compute_foreground_masks(rgb.current_frame)
    
        # update rgb state of each pixel according to the FSM
        bg_predic = rgb.update_detection_state()
    
        # Extract the bounding box from the obtained background and store them into a DataFame
        rgb.extract_proposal_bbox(model)
        
        # Manage the DATABASE 
        rgb_proposal_bbox = rgb.process_proposal_bbox(database_path, folder_path)
        
        # Draw the bbox on the original image
        result = rgb.current_frame.copy()
        draw_bounding_box_df(result, rgb_proposal_bbox, agg_lim = AGG_THRESH)
    
        
        ############### Plot result #####################
        
        # images = [result, cv2.applyColorMap(rgb.background_state*50, cv2.COLORMAP_HSV), cv2.applyColorMap( bg_predic.astype(np.uint8)*60, cv2.COLORMAP_HSV ),
        #         rgb.proposal_mask, rgb.foreground_mask_long_term, rgb.foreground_mask_short_term]
        # legend = ["result_lst", "rgb_state", "bg_pred", "prop_bg", "fg_m_l", "fg_m_s"]
        # im = import_images.create_6tuple(images, legend, "output_video_Medium")
        # cv2.imshow("Result", im)
            
        cv2.imshow('Video', result)
        k = cv2.waitKey(1)
        if k == 27 or k == ord("q"):   # keyboard Escape
            fps.stop()
            cv2.destroyAllWindows()
            cap.release() 
            sys.exit()
       
        fps.update()

