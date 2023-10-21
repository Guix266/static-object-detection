# -*- coding: utf-8 -*-
"""
Main program to treat images from a video.
"""

import numpy as np
import time
import sqlite3
import cv2
import tensorflow as tf

from luggage_detection import *
import import_images
import sequential_bg


# =============================================================================
#     Import images
# =============================================================================
### AVSS
chosen = 2
lst = ["AVSS_AB_Easy_Divx", "AVSS_AB_Medium_Divx", "AVSS_AB_Hard_Divx"]
num = [[251, 5450], [251, 4825], [251, 5300]]
video_name = "Video_dataset/"+ lst[chosen]+".avi"
n_first_image, n_last_image = num[chosen]

### PETS
# n_serie = 1  #serie of video selected
# picture_name = "S1-T1-C."
# folder = "Video_dataset/S1-T1-C/video/pets2006/S1-T1-C/"+str(n_serie)


step = 10
# s1 = import_images.load_series(picture_name, folder, n_first_image, n_last_image, step = step)
s1 = import_images.load_video(video_name = video_name, inf = n_first_image, sup = n_last_image, step = step)
# import_images.plot_video(s1, interval=40, grey = False, title = "video")

# =============================================================================
#     Sequential background creation
# =============================================================================
p = step*4
N_seq = 40

# bg = cv2.imread("Video_dataset/background.jpg")
print("[INFO] : background initialization...")
bg = sequential_bg.sequential_bg_init(s1[0 : len(s1) : p][:N_seq])
# import_images.plot_im([bg], ["Sequential Background"], binary = False)
s1 = [bg]*40 + s1


# =============================================================================
#     Import CNN
# =============================================================================

model = tf.keras.models.load_model('CNN/models/pretrain_cnn.h5')

# =============================================================================
# Connection BDD
# =============================================================================
#%%
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
# reset_db(database_path)

# =============================================================================
#     Algorithme
# =============================================================================

t1 = time.perf_counter()

fg_m_l = []
fg_m_s = []
rgb_state = []
prop_bg = []
result_lst = []
bg_l_images = []
bg_s_images = []
bg_pred = []


# cv2.namedWindow("result",cv2.WINDOW_NORMAL)
# cv2.resizeWindow("result", 1300,800)

# Create instance
rgb = LuggageDetection( s1[0] )
# j = 0
# k = 0
for i in range(len(s1) - 1):
    print("\r[INFO] : processing of the image {}...".format(i), end='')
    # print('============================================================')
    # print("[INFO] : processing of the image {}...".format(i))

    # get rgb dual background (long and short sensitivity)
    # N.B. background is black (0) and foreground white (1)
    ab, ab2, c, a = rgb.compute_foreground_masks(rgb.current_frame)

    # update rgb state of each pixel according to the FSM
    bg_predic = rgb.update_detection_state()

    # Extract the bounding box from the obtained background and store them into a DataFame
    rgb.extract_proposal_bbox(model)
    
    # Manage the DATABASE 
    rgb_proposal_bbox = rgb.process_proposal_bbox(database_path, folder_path)
    
    # Draw the bbox on the original image
    result = rgb.current_frame.copy()
    draw_bounding_box_df(result, rgb_proposal_bbox, agg_lim = AGG_THRESH)

    # print(rgb_proposal_bbox)
    
    ############### Plot result #####################
    
    if i >= 190:
        
        
    #     images = [result, cv2.applyColorMap(rgb.background_state*50, cv2.COLORMAP_HSV), cv2.applyColorMap( bg_predic.astype(np.uint8)*60, cv2.COLORMAP_HSV ),
    #               rgb.proposal_mask, rgb.foreground_mask_long_term, rgb.foreground_mask_short_term]
    #     legend = ["result_lst", "rgb_state", "bg_pred", "prop_bg", "fg_m_l", "fg_m_s"]
        
    #     im = import_images.create_6tuple(images, legend, "output_video_Medium")
    #     cv2.imshow("Result", im)
        
        cv2.imshow("AFTER_postprocessing", rgb.foreground_mask_short_term*255)
        cv2.imshow("After med morpho", ab2.astype(np.uint8)*255)
        cv2.imshow("BASIC", ab.astype(np.uint8)*255)
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            cv2.destroyAllWindows()
            break

    ################## SAVE IMAGES ####################

    # for box in rgb_proposal_bbox.itertuples():

    #     # Save the images in a folder
    #     crop_im = rgb.current_frame[box.POS_Y:box.POS_Y+box.HEIGHT, box.POS_X:box.POS_X+box.WIDTH]
    #     if box.AGG_CLASS >= 10:
    #         name = "valise_"+str(j)+".jpg"
    #         cv2.imwrite("Output/images/valise/"+name, crop_im)
    #         j = j+1
    #     else:
    #         name = "other_"+str(k)+".jpg"
    #         cv2.imwrite("Output/images/other/"+name, crop_im)
    #         k = k+1
        
    ########################################
    
    # Update for next image
    rgb.current_frame = s1[ i+1 ]
    

    # Add the images to lists to print them
    prop_bg.append(rgb.proposal_mask)
    fg_m_l.append(rgb.foreground_mask_long_term)
    fg_m_s.append(rgb.foreground_mask_short_term)
    rgb_state.append(cv2.applyColorMap(rgb.background_state*50, cv2.COLORMAP_HSV))
    result_lst.append(result)
    bg_pred.append(cv2.applyColorMap( bg_predic.astype(np.uint8)*60, cv2.COLORMAP_HSV ))

    bg_l_images.append(rgb.f_bg_long.getBackgroundImage())
    bg_s_images.append(rgb.f_bg_short.getBackgroundImage())

t2 = int((time.perf_counter() - t1)*10)/10
print("\n[INFO] : video processed in t = {} seconds".format(t2))


# =============================================================================
#  Plot the result
# =============================================================================

# Create a video with up to 6 streams
images = [result_lst, rgb_state, bg_pred, prop_bg, fg_m_l, fg_m_s]
legend = ["result_lst", "rgb_state", "bg_pred", "prop_bg", "fg_m_l", "fg_m_s"]
import_images.create_video6(images, legend, "output_video_"+lst[chosen])

print("[INFO] : output video created")

# =============================================================================
# %%
# =============================================================================
print_db(database_path)

# if __name__ == "__main__":
#     luggage_detection()

 