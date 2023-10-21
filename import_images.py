# -*- coding: utf-8 -*-

"""
This module contains different functions to import and save images.
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

def load_im(image, folder = "", grey = False):
    """Load an image (grey for grey values)"""
    if grey:
        image = cv2.imread(os.path.join(folder,image), cv2.IMREAD_GRAYSCALE)
    else:
        image = cv2.imread(os.path.join(folder,image), cv2.IMREAD_COLOR)
    return(image)


def plot_im(images, legend = 0, binary = False):
    """plot the images in a window.
    The images are in a list"""
    i = len(images)
    fig, axs = plt.subplots(nrows = 1, ncols = i, figsize = (20,6))
    for j in range(i):
        ax = plt.subplot(1,i, j+1)
        ax.set_xticks([])
        ax.set_yticks([])
        if type(legend) == list:
            if len(legend) == i:
                plt.title(legend[j], fontsize=24)
            else:
                return("Error: Wrong number of elements in the legend")
        if binary:
            plt.imshow(images[j],  cmap = 'gray')
        else:
            plt.imshow(images[j])
    plt.show()
    return(fig)


def load_series(picture_name, folder, inf, sup, step = 1, grey = False):
    """ Load the PETS image serie
    Inf, sup from 0 to 1000"""
    print("[INFO] : video importation...")
    im = []
    for i in range(inf, sup, step):
        num = str(i)
        while len(num) < 5:
            num = "0"+num
        im.append(load_im(picture_name+num+".jpeg", folder, grey = grey))
    return(im)

def load_video(video_name, folder = "", inf = 1000, sup = 5000, step = 1, grey = False) :
    """ Load a video as a list of consecutive images from inf to sup """
    print("[INFO] : video importation...")
    im = []
    cap = cv2.VideoCapture(video_name)
    count = 0
    while(cap.isOpened() and count < sup):
        ret, frame = cap.read()   
        if count%step == 0 and count >= inf :
            if grey == True :
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(frame, (720,576), interpolation = cv2.INTER_AREA)
            im.append(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        count += 1
    cap.release()
    cv2.destroyAllWindows()
    return(im)


def create_video6(images, legend, name) :
    """ Create a video with up to 6 streams """
    a = (len(images)-1)//3 + 1
    b = (len(images)-1)%3 + 1
    if len(images) >= 4 :
        b = 3

    if len(images)>6 :
        print("To many video streams")
        return 0
    
    height, width = images[0][0].shape[:2]
    out = cv2.VideoWriter(os.path.join('Output',name +'.avi'),cv2.VideoWriter_fourcc(*'DIVX'), 5, (b*(width+10),a*(height+10)))
    new_image = np.ones([a*(height+10), b*(width+10), 3], dtype=np.uint8) * 250
    for i in range(len(images[0])) :
        for j in range(len(images)) :
            i2 = j//3
            j2 = j%3
            if images[j][i].shape[-1] != 3 :
                new_image[i2*height + i2*10 : (i2+1)*height + i2*10, j2*width + j2*10 : (j2+1)*width + j2*10] = cv2.cvtColor(images[j][i].astype(np.uint8)*255, cv2.COLOR_GRAY2BGR)
            else : 
                new_image[i2*height + i2*10 : (i2+1)*height + i2*10, j2*width + j2*10 : (j2+1)*width + j2*10] = images[j][i]
            text_org = (int((j2+0.05)*width), int((i2+0.95)*height+10))
        new_image = cv2.putText(new_image, legend[j], text_org, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,80,80), thickness=4)
        out.write(new_image)
    out.release()

def create_6tuple(images, legend, name):
    """ Create an image with 6 images and add a legend """
    a = (len(images)-1)//3 + 1
    b = (len(images)-1)%3 + 1
    if len(images) >= 4 :
        b = 3

    if len(images)>6 :
        print("To many video streams")
        return 0
    
    height, width = images[0].shape[:2]
    new_image = np.ones([a*(height+10), b*(width+10), 3], dtype=np.uint8) * 250
    for j in range(len(images)) :
        i2 = j//3
        j2 = j%3
        if images[j].shape[-1] != 3 :
            new_image[i2*height + i2*10 : (i2+1)*height + i2*10, j2*width + j2*10 : (j2+1)*width + j2*10] = cv2.cvtColor(images[j].astype(np.uint8)*255, cv2.COLOR_GRAY2BGR)
        else : 
            new_image[i2*height + i2*10 : (i2+1)*height + i2*10, j2*width + j2*10 : (j2+1)*width + j2*10] = images[j]
        text_org = (int((j2+0.05)*width), int((i2+0.95)*height+10))
        new_image = cv2.putText(new_image, legend[j], text_org, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,80,80), thickness=4)
    return(new_image)