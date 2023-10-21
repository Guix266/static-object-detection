# -*- coding: utf-8 -*-

"""
This module contains the functions to initialise the background sequentially.
"""

import numpy as np
import cv2
import math

import import_images


# =============================================================================
#     Constants and functions
# =============================================================================

N = 16              # block size
a = 0.3             # cost_function paramerter
threshold = 30      # threshold
ratio = 0.8         # images ratio where we are sure it is background

def similar(b, r) : 
    C1 = cv2.subtract(b, r)   #if b[i]<r[i] return 0
    C2 = cv2.subtract(r, b)
    retval1, pos1 = cv2.checkRange(C1, maxVal = threshold)
    retval2, pos2 = cv2.checkRange(C2, maxVal = threshold)
    return (retval1 and retval2)

def cost_function(Wk, a, frame1, frame2, n_images) : 
    res = 0
    wk = Wk / n_images
    C = cv2.dct(frame1)
    D = cv2.dct(frame2)
    E = abs(cv2.add(C, D))
    res = cv2.sumElems(E)[0]
    return res * math.exp(-(a * wk))


# =============================================================================
#     Principal function
# =============================================================================

def sequential_bg_init(images_RGB) : 
    images = []
    for image_RGB in images_RGB :
        images.append(cv2.cvtColor(image_RGB, cv2.COLOR_BGR2GRAY))

    W = images[0].shape[0]   # row number
    H = images[0].shape[1]   # column number
    n_images = len(images)   # number of images


    # =============================================================================
    #     Collection of block representatives
    # =============================================================================

    R = np.zeros([n_images, W//N, H//N, N, N], dtype=np.uint8)      # set of grey frames
    R_RGB = np.zeros([n_images, W//N, H//N, N, N, 3], dtype=np.uint8)      # set of RGB frames
    P = np.zeros([n_images, W//N, H//N])                                # weights matrice

    for f, imagef in enumerate(images) :
        ### Split input frame If into blocks, each with a size of N × N.
        for i in range(W//N) : 
            for j in range (H//N) : 
                b = imagef[i*N:(i+1)*N, j*N:(j+1)*N]
                b_RGB = images_RGB[f][i*N:(i+1)*N, j*N:(j+1)*N]
                rm = []
                for m in range(f) :
                    r = R[m, i, j]
                    if similar(b, r) == True :
                        rm = r
                        m1 = m
                if rm != [] : 
                    R[m1, i, j] = cv2.addWeighted(R[m1, i, j], P[m1, i, j]/(P[m1, i, j] + 1), b, 1/(P[m1, i, j] + 1), 0)
                    R_RGB[m1, i, j] = cv2.addWeighted(R_RGB[m1, i, j], P[m1, i, j]/(P[m1, i, j] + 1), b_RGB, 1/(P[m1, i, j] + 1), 0)
                    P[m1, i, j] += 1
                else :
                    R[f, i, j] = b.copy()
                    R_RGB[f, i, j] = b_RGB.copy()
                    P[f, i, j] = 1


    # =============================================================================
    #     Partial background reconstruction
    # =============================================================================

    BG = - np.ones(images[0].shape, dtype = np.int16)
    BG_RGB = np.zeros(images_RGB[0].shape, dtype=np.uint8)
    for i in range(W//N) :
        for j in range(H//N) :
            if P[0, i, j] >= int(n_images * ratio) :
                BG[i*N:(i+1)*N, j*N:(j+1)*N] = R[0, i, j]
                BG_RGB[i*N:(i+1)*N, j*N:(j+1)*N] = R_RGB[0, i, j]
    # import_images.plot_im([BG], binary = True)
    # import_images.plot_im([BG_RGB], binary = False)


    # =============================================================================
    #     Estimation of the missing background
    # =============================================================================

    ##### We fill the background with 3 neighbours
    loop = 0
    while BG.min() < 0 and loop <= 100:  #while BG is not full
        loop += 1
        for i in range(W//N) :
            for j in range(H//N) :
                if BG[i*N:(i+1)*N, j*N:(j+1)*N].min() < 0 :   #empty block
                    V = - np.ones([3, 3, N, N])  # neighbours
                    C = np.zeros([2*N, 2*N])     #superblock
                    min_D = np.zeros([2*N, 2*N])  
                    min_frame_RGB = np.zeros([N, N, 3], dtype=np.uint8)  
                    min_cost = float('inf')
                    # neighbours count :
                    # |(0,2)|(1,2)|(2,2)|
                    # |(0,1)|(1,1)|(2,1)|
                    # |(0,0)|(1,0)|(2,0)|
                    # add neighbours to V
                    for i2 in [-1, 0, 1] :
                        for j2 in [-1, 0, 1] :
                            if i+i2 >= 0 and i+i2 < W//N and j+j2 >= 0 and j+j2 < H//N and BG[(i+i2)*N : (i+i2+1)*N, (j+j2)*N : (j+j2+1)*N].min() >= 0 :
                                V[i2+1, j2+1] = BG[(i+i2)*N : (i+i2+1)*N, (j+j2)*N : (j+j2+1)*N]
                    
                    has_3_neighbours = False
                    #### Case in angle
                    #case 1 = top left
                    if V[0,1].min() >= 0 and V[0,2].min() >= 0 and V[1,2].min() >= 0 and has_3_neighbours == False :
                        C[0:N, 0:N]     = V[0,1]
                        C[0:N, N:2*N]   = V[0,2]
                        C[N:2*N, N:2*N] = V[1,2]
                        # C[N:2*N, 0:N]   =
                        has_3_neighbours = True
                        i3 = 1
                        j3 = 0
                    #case 2 = top right
                    if V[1,2].min() >= 0 and V[2,2].min() >= 0 and V[2,1].min() >= 0 and has_3_neighbours == False :
                        # C[0:N, 0:N]     =
                        C[0:N, N:2*N]   = V[1,2]
                        C[N:2*N, N:2*N] = V[2,2]
                        C[N:2*N, 0:N]   = V[2,1]
                        has_3_neighbours = True
                        i3 = 0
                        j3 = 0
                    #case 3 = bottom right
                    if V[2,1].min() >= 0 and V[2,0].min() >= 0 and V[1,0].min() >= 0 and has_3_neighbours == False :
                        C[0:N, 0:N]     = V[1,0]
                        # C[0:N, N:2*N]   =
                        C[N:2*N, N:2*N] = V[2,1]
                        C[N:2*N, 0:N]   = V[2,0]
                        has_3_neighbours = True
                        i3 = 0
                        j3 = 1
                    #case 4 = bottom left
                    if V[1,0].min() >= 0 and V[0,0].min() >= 0 and V[0,1].min() >= 0 and has_3_neighbours == False :
                        C[0:N, 0:N]     = V[0,0]
                        C[0:N, N:2*N]   = V[0,1]
                        # C[N:2*N, N:2*N] =
                        C[N:2*N, 0:N]   = V[1,0]
                        has_3_neighbours = True
                        i3 = 1
                        j3 = 1
                    D = C.copy()
                    if has_3_neighbours == True :   # enough neighbours to estimate the background
                        for k in range(n_images) :
                            if P[k, i, j] > 0 :
                                D[i3*N : (i3+1)*N, j3*N : (j3+1)*N] = R[k, i, j].copy()
                                cost = cost_function(P[k, i, j], a, C, D, n_images)
                                if cost < min_cost :
                                    min_cost = cost
                                    min_D = D.copy()
                                    min_frame_RGB = R_RGB[k, i, j].copy()
                        BG[i*N:(i+1)*N, j*N:(j+1)*N] = min_D[i3*N : (i3+1)*N, j3*N : (j3+1)*N].copy()
                        BG_RGB[i*N:(i+1)*N, j*N:(j+1)*N] = min_frame_RGB.copy()
    # import_images.plot_im([BG], binary = True)
    # import_images.plot_im([BG_RGB], binary = False)

    ##### If the background is not full we fill it with 1 neighbour
    loop = 0
    while BG.min() < 0 and loop <= 100:  #while BG is not full
        loop += 1
        for i in range(W//N) :
            for j in range(H//N) :
                if BG[i*N:(i+1)*N, j*N:(j+1)*N].min() < 0 :   #empty block
                    V = - np.ones([3, 3, N, N])  # neighbours
                    C = np.zeros([2*N, 2*N])     #superblock
                    min_D = np.zeros([2*N, 2*N]) 
                    min_frame_RGB = np.zeros([N, N, 3], dtype=np.uint8)   
                    min_cost = float('inf')
                    # neighbours count :
                    # |(0,2)|(1,2)|(2,2)|
                    # |(0,1)|(1,1)|(2,1)|
                    # |(0,0)|(1,0)|(2,0)|
                    # add neighbours to V
                    for i2 in [-1, 0, 1] :
                        for j2 in [-1, 0, 1] :
                            if i+i2 >= 0 and i+i2 < W//N and j+j2 >= 0 and j+j2 < H//N and BG[(i+i2)*N : (i+i2+1)*N, (j+j2)*N : (j+j2+1)*N].min() >= 0 :
                                V[i2+1, j2+1] = BG[(i+i2)*N : (i+i2+1)*N, (j+j2)*N : (j+j2+1)*N]
                    
                    has_1_neighbours = False
                    #case 1 = top
                    if V[1,2].min() >= 0 and has_1_neighbours == False :
                        C[int(N/2):int(N+N/2), N:2*N] = V[1,2]
                        has_1_neighbours = True
                        i3 = 1/2
                        j3 = 0
                    #case 2 = right
                    if V[2,1].min() >= 0 and has_1_neighbours == False :
                        C[N:2*N, int(N/2):int(N+N/2)]   = V[2,1]
                        has_1_neighbours = True
                        i3 = 0
                        j3 = 1/2
                    #case 3 = bottom
                    if V[1,0].min() >= 0 and has_1_neighbours == False :
                        C[int(N/2):int(N+N/2), 0:N]     = V[1,0]
                        has_1_neighbours = True
                        i3 = 1/2
                        j3 = 1
                    #case 4 = left
                    if V[0,1].min() >= 0 and has_1_neighbours == False :
                        C[0:N, int(N/2):int(N+N/2)]   = V[0,1]
                        has_1_neighbours = True
                        i3 = 1
                        j3 = 1/2
                    D = C.copy()
                    if has_1_neighbours == True :   # enough neighbours to estimate the background
                        for k in range(n_images) :
                            if P[k, i, j] > 0 :
                                D[int(i3*N) : int((i3+1)*N), int(j3*N) : int((j3+1)*N)] = R[k, i, j].copy()
                                cost = cost_function(P[k, i, j], a, C, D, n_images)
                                if cost < min_cost :
                                    min_cost = cost
                                    min_D = D.copy()
                                    min_frame_RGB = R_RGB[k, i, j].copy()
                        BG[i*N:(i+1)*N, j*N:(j+1)*N] = min_D[int(i3*N) : int((i3+1)*N), int(j3*N) : int((j3+1)*N)].copy()
                        BG_RGB[i*N:(i+1)*N, j*N:(j+1)*N] = min_frame_RGB.copy()
    # import_images.plot_im([BG], binary = True)
    # import_images.plot_im([BG_RGB], binary = False)

    ##### If the background is still not full we fill it with the first image
    for i in range(W) :
        for j in range(H) : 
            if   BG[i, j] == -1 :
                BG[i, j] = images[0][i, j]
                BG_RGB[i, j] = images_RGB[0][i, j]

    return BG_RGB

if __name__ == "__main__":

    # =============================================================================
    #     Import images
    # =============================================================================

    n_serie = 4  #serie of video selected
    picture_name = "S1-T1-C."
    video_name = "Datasets/AVSS_AB_Medium_Divx.avi"
    folder = "S1-T1-C/video/pets2006/S1-T1-C/"+str(n_serie)
    n_first_image = 800
    n_last_image  = 5000
    step          = 100  
    # s = import_images.load_series(picture_name, folder, 0, 3000, step = step, grey = False)
    s = import_images.load_video(video_name = video_name, inf = n_first_image, sup = n_last_image, step = step)
    s = s[:40]
    BG_RGB = sequential_bg_init(s)
    import_images.plot_im([BG_RGB], binary = False)
