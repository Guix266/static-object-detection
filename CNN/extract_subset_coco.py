# -*- coding: utf-8 -*-

"""
This module can be used to import coco images and preprocess them to train our CNN
"""

from pycocotools.coco import COCO
import os
import pandas as pd
from PIL import Image
import random


# =============================================================================
#  COCO DATASET HAS THE FOLLOWING CLASSES :
# =============================================================================
# {0: u'__background__',
#  1: u'person',
#  2: u'bicycle',
#  3: u'car',
#  4: u'motorcycle',
#  5: u'airplane',
#  6: u'bus',
#  7: u'train',
#  8: u'truck',
#  9: u'boat',
#  10: u'traffic light',
#  11: u'fire hydrant',
#  12: u'stop sign',
#  13: u'parking meter',
#  14: u'bench',
#  15: u'bird',
#  16: u'cat',
#  17: u'dog',
#  18: u'horse',
#  19: u'sheep',
#  20: u'cow',
#  21: u'elephant',
#  22: u'bear',
#  23: u'zebra',
#  24: u'giraffe',
#  25: u'backpack',
#  26: u'umbrella',
#  27: u'handbag',
#  28: u'tie',
#  29: u'suitcase',
#  30: u'frisbee',
#  31: u'skis',
#  32: u'snowboard',
#  33: u'sports ball',
#  34: u'kite',
#  35: u'baseball bat',
#  36: u'baseball glove',
#  37: u'skateboard',
#  38: u'surfboard',
#  39: u'tennis racket',
#  40: u'bottle',
#  41: u'wine glass',
#  42: u'cup',
#  43: u'fork',
#  44: u'knife',
#  45: u'spoon',
#  46: u'bowl',
#  47: u'banana',
#  48: u'apple',
#  49: u'sandwich',
#  50: u'orange',
#  51: u'broccoli',
#  52: u'carrot',
#  53: u'hot dog',
#  54: u'pizza',
#  55: u'donut',
#  56: u'cake',
#  57: u'chair',
#  58: u'couch',
#  59: u'potted plant',
#  60: u'bed',
#  61: u'dining table',
#  62: u'toilet',
#  63: u'tv',
#  64: u'laptop',
#  65: u'mouse',
#  66: u'remote',
#  67: u'keyboard',
#  68: u'cell phone',
#  69: u'microwave',
#  70: u'oven',
#  71: u'toaster',
#  72: u'sink',
#  73: u'refrigerator',
#  74: u'book',
#  75: u'clock',
#  76: u'vase',
#  77: u'scissors',
#  78: u'teddy bear',
#  79: u'hair drier',
#  80: u'toothbrush'}



def get_bounding(category, save_folder, M):
    """
    Save in caption.csv the bounding boxes parameters of the differents categories
    It is poddible to choose to only take the first M images
    ==========================================================================
    :param category: list of the CLASSES to import
    :param save_folder: path of the folder where to save the images
    :param M: max number of images to save
    """
    
    # note this only refers to  the training set and not the validation set
    coco = COCO('coco/annotations/instances_train2017.json')  
    
    # note this only refers to the captions of the training set and not the validation set   
    caps = COCO('coco/annotations/captions_train2017.json') 
    
    categories = coco.loadCats(coco.getCatIds())
    names = [cat['name'] for cat in categories] 
    
    print("Available categories: ")
    for index, n in enumerate(names):
        print(index, n)
    
    category_ids = coco.getCatIds(catNms=[category])
    image_ids = coco.getImgIds(catIds=category_ids)
    images = coco.loadImgs(image_ids)
    annIds = caps.getAnnIds(imgIds=image_ids)
    annotations = caps.loadAnns(annIds)
    annIds2 = coco.getAnnIds(catIds= category_ids)
    annotations2 = coco.loadAnns(annIds2)
    
    # Split the annotations every 5 captions since there are 5 captions for each image
    annotations = [annotations[x:x + 5] for x in range(0, len(annotations), 5)]
    
    # Create empty dataframe with two columns for the image file name and the corresponding captions
    # df = pd.DataFrame(columns=['image_id', 'caption'])
    df = pd.DataFrame(columns=['image_id', 'x_topleft', "y_topleft", "bbox_width", "bbox_height"])
    
    # Create folder in for the images of the selected category
    if not(os.path.isdir(save_folder + category)):
        os.mkdir(save_folder+category)
    
    # Create map for image id (key) to captions (values)
    captions_dict = {}
    for i, n in enumerate(annotations):
        captions_dict[annotations[i][0]['image_id']] = annotations[i]
    
    print("[INFO] : Create the csv...")
    j = 0
    horse_file_names = []
    for img in images:
        # print("\r {}".format(j))
        horse_file_names.append(img['file_name'])
        x_topleft   = 0
        y_topleft   = 0
        bbox_width  = 0
        bbox_height = 0
        bboxs = []
        for ann in annotations2 :
            if ann['image_id'] == img['id'] :
                x_topleft   = ann['bbox'][0]
                y_topleft   = ann['bbox'][1]
                bbox_width  = ann['bbox'][2]
                bbox_height = ann['bbox'][3]
                bboxs.append([x_topleft, y_topleft, bbox_width, bbox_height])
        for bbox in bboxs :
        # bbox = bboxs[0]
            df.loc[len(df)] = [img['file_name'], bbox[0], bbox[1], bbox[2], bbox[3]]
            j += 1
        if j >= M:
            break
    
    # Save category
    df.to_csv(save_folder+category + "/captions.csv", index=False)
    
   

def resize_im(n = 0, size = (64,64), add = 15):
    """
    Crop the in-context images to only keep the desired object
    ==========================================================================
    :param n: max number of images to save
    :param size : size of the output images, correspond to the input size of the CNN
    :add: if not 0 represent the maximum number of pixels to augment the size of the 
            bounding box to crop
    """
    df = pd.read_csv(save_folder+category + "/captions.csv")
    if n == 0:
        n, m = df.shape
    for i in range(n):
        print("\r[INFO] : Save the image {}".format(i), end='')
        im_name = df.iloc[i, 0]
        x_topleft   = df.iloc[i, 1]
        y_topleft   = df.iloc[i, 2]
        bbox_width  = df.iloc[i, 3]
        bbox_height = df.iloc[i, 4]

        im = Image.open(os.path.join(coco_image_repertory,im_name))
        
        # Crop the image
        width, height = im.size
        r1 = min( x_topleft, random.randrange(add))
        r2 = min( y_topleft, random.randrange(add))
        r3 = min( width - x_topleft - bbox_width, random.randrange(add))
        r4 = min( height - y_topleft - bbox_height, random.randrange(add))
        im1 = im.crop((x_topleft-r1, y_topleft-r2, x_topleft+bbox_width+r3 , y_topleft+bbox_height+r4)) 
        
        # Resize it
        im2 = im1.resize(size, resample=Image.BILINEAR)
        
        # Save it in the folder
        num = str(i)
        while len(num) < 5:
            num = "0"+num
        im2.save(os.path.join(save_folder+category, category + "_" + num +".jpg"))


def new_size(folder2, category, size = (64, 64)):
    """
    rezise the images if need
    """
    
    # Create folder in for the images of the selected category
    lst = os.listdir("data_subsets/bounding_boxes/")
    # print(lst)
    if category not in lst:
        os.mkdir(save_folder+category)
    
    for filename in os.listdir("data_subsets/"+folder2):
        print("\r[INFO] : processing of the image "+filename, end='')
        if filename[-1] == 'g':
            im = Image.open(os.path.join("data_subsets/"+folder2, filename))
            
            # Greyscale
            im = im.convert('L')
            
            # Resize
            im1 = im.resize(size, resample=Image.BILINEAR)
            
            im1.save(os.path.join("data_subsets/model_data/"+ category, filename))
            # break
        

######################################################################
### Class to import
######################################################################

### Mode d'emploi :
#1)  Importer cocodatasets : train_set ou validation_set
#    Importer les annotations et repporter le path des fichier .json dans get_bounding

#2) Adapter les paths ci-dessous

#3) Changer les paramètres : CLASS, size et n

#4) Utiliser le programme


CLASS = ['backpack', 'suitcase', 'handbag', 'bench', 'chair', 'person', 
         'train', 'bicycle', 'motorcycle', 'bus', 'truck', 'car', 'fire hydrant' ]


n = 0                   # Nb images importées max (0 = no limits)
size = (64,64)          # Taille voulue des images

save_folder = 'data_subsets/train2017/'             #Where to save the images
coco_image_repertory = 'coco/train2017/train2017'   #Where are the coco images


for category in CLASS:
    
    # Save in caption.csv the bounding boxes and image name (MAX : 10 000 images)
    get_bounding(category, save_folder, n)
    
    # Crop the n first images of the class + save them into folder_name
    resize_im(n = n, size = size, add = 4)
    
    # Resize the images to get the model data
    # new_size(folder_name, category, size = size)

