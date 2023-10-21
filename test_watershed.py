"""
This module contains the functions for image processing and foreground detection
"""

import numpy as np
import cv2
import import_images

from const import *
from bg_models import *
from luggage_detection import *

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

cv2.imshow("im", s1[190])
cv2.waitKey(0)



