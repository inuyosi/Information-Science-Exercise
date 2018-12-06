# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt

#####################################################
## generate 'm' level Gussian pyramid of image 'G'
# def generateGussianPyr(G, m):
#
## generate 'm' level Gussian pyramid of image 'G0'
# def generateLaplacianPyr(G0, m):
#
## generate 'm' level Laplacian pyramid of image 'lp'
# def reconstructLaplacianPyr(lp, m):
#
## show image pyramid
# def showPyr(pyr):
#####################################################


# generate 'm' level Gussian pyramid of image 'G'
def generateGussianPyr(G, m):
    gp = [G]
    for i in range(m-1):
        G = cv2.pyrDown(G)
        gp.append(G)
    return gp

# generate 'm' level Gussian pyramid of image 'G0'
def generateLaplacianPyr(G0, m):
    lp = []
    for i in range(m-1):
        size = (G0.shape[1], G0.shape[0])
        G1 = cv2.pyrDown(G0)
        tmpG = cv2.pyrUp(G1, dstsize = size)
        lp.append(cv2.subtract(G0, tmpG))
        G0 = G1
    lp.append(G1)
    return lp

# generate 'm' level Laplacian pyramid of image 'lp'
def reconstructLaplacianPyr(lp, m):
    rImg = lp[m-1]
    for i in range(m-1,0,-1):
        size = (lp[i-1].shape[1], lp[i-1].shape[0])
        lpitmp = cv2.pyrUp(rImg, dstsize = size)
        rImg = cv2.add(lpitmp, lp[i-1])
    return rImg

# show image pyramid
def showPyr(pyr):
    m = len(pyr)
    h = np.round(np.sqrt(m))
    w = np.ceil(np.sqrt(m))
    for i in range(m):
        if pyr[i].min()<0:
            pyr[i] *= 1.5 # enhance Laplacian values
        plt.subplot(h,w,i+1)
        if len(pyr[i].shape) == 3: # color
            b,g,r = cv2.split( np.abs(pyr[i]) )
            plt.imshow( cv2.merge((r,g,b)) ) # cv2:BGR <--> plt:RGB
        else: # gray
            plt.imshow( np.abs(pyr[i]), 'gray' )
    plt.show()


def showMultiImage(wname, imgs):
    for i in range(len(imgs)):
        plt.subplot(131+i)
        plt.imshow(cv2.cvtColor(np.uint8(myNormalization(imgs[i])*255),cv2.COLOR_BGR2RGB))
        plt.axis('off')
    plt.show()

def showImage(wname, img, mode):
    if mode:
        img = np.uint8(myNormalization(img)*255)
    else:
        imgMean = np.mean(img)
        if imgMean < 1:
            img = img*255
        img = np.uint8(img)

    cv2.namedWindow(wname, cv2.WINDOW_KEEPRATIO | cv2.WINDOW_NORMAL)
    cv2.imshow(wname, img)
#    cv2.waitKey(0)


def writeImage(fname, img, mode):
    if mode:
        img = np.uint8(myNormalization(img)*255)
    else:
        imgMean = np.mean(img)
        if imgMean < 1:
            img = img*255
        img = np.uint8(img)

    cv2.imwrite(fname, img)


def myNormalization(img):
    imgMax = np.percentile(img,99.9)
    imgMin = np.percentile(img,3.0)
    if imgMax-imgMin == 0:
        return img
    img2 = (img-imgMin)/(imgMax-imgMin)
    img2[img2<0] = 0
    img2[img2>1] = 1
    return img2
