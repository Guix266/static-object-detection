# -*- coding: utf-8 -*-

"""
This module contains different functions used in the program.
"""

import numpy as np

# =============================================================================
#  Input Images constants
# =============================================================================
step = 10
frames_per_seconds = 25                 # in the original video
ratio = frames_per_seconds / step       # frames processed per second
ratio = 1                               # frames processed per second


# =============================================================================
#  Background substraction constants 
# =============================================================================
### Dual Background Porikli method using Zivkovich model background for (RGB) Parameters
BG_ZIV_LONG_LRATE = 0.0005 / ratio        #: Background learning rate in Zivkovich method for long background model, original = 0.0005
BG_ZIV_SHORT_LRATE = 0.005 / ratio        #: Background learning rate in Zivkovich method for short background model, original = 0.02
BG_ZIV_HIST = 1                           #: History for Zivkovick background method

BG_ZIV_LONG_THRESH = 50        #: Threshold for Zivkovich method for long background model
BG_ZIV_SHORT_THRESH = 50       #: Threshold for Zivkovich method for short background model


# =============================================================================
#  FSM MODEL CONSTANTS
# =============================================================================
BG = 0       #backgroung
MF = 1       #Moving Foreground
CSF = 2      #Candidate Static Foreground
OCSF = 3     #Occluded Static Static Foreground
SFO = 4      #Static Foreground Object
states = [ BG, MF, CSF, OCSF, SFO]
                        #  BG       MF      CSF     OCSF    SFO       # LS  
loockup_table = np.array([[BG,      BG,      BG,     BG,    SFO ],    # 00
                          [MF,      CSF,    CSF,    CSF,    SFO ],    # 10
                          [BG,      BG,    OCSF,   OCSF,    SFO ],    # 01
                          [MF,      MF,    OCSF,   OCSF,    SFO ]],   # 11
                          dtype=np.int8)

FMI_BG_DIF = int(15 * ratio)    # Number for the test from CSF to SF0 (15 seconds)


# =============================================================================
#  Bounding box processing
# =============================================================================
BBOX_MIN_AREA = 250             #: minimum area in pixel to create a bounding box

# BBOX AGGREGATOR
AGG_MAX = 20
AGG_MIN = -15
AGG_THRESH = 10


# =============================================================================
#  CNN CONSTANTS
# =============================================================================
CLASS_THRESH = 0.7              #: Threshold for the CNN classification

CLASSES = ['backpack', 'suitcase', 'handbag', 'bench', 'chair', 'person', 
          'train', 'bicycle', 'motorcycle', 'bus', 'truck', 'car', 'fire hydrant' ]
