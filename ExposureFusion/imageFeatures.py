import numpy as np
import cv2
import matplotlib.pyplot as plt
import pySaliencyMap
import pyrFunc as pf


def contrast(imgs):  # absolute Laplacian value
    n = len(imgs)
    h, w, c = imgs[0].shape[:3]
    C = np.zeros((h, w, n))
    for i in range(n):
        mono = rgb2gray(imgs[i])
        C[:, :, i] = abs(cv2.Laplacian(mono, cv2.CV_64F, ksize=1))

#    showImage('Contrast', C, 0)
    return C


def saturation(imgs):  # standard deviation of color (R,G,B)
    n = len(imgs)
    h, w, c = imgs[0].shape[:3]
    S = np.zeros((h, w, n))
    for i in range(n):
        rgb = imgs[i]
        mu = (rgb[:, :, 0]+rgb[:, :, 1]+rgb[:, :, 2])/3
        S[:, :, i] = np.sqrt((np.power(
            rgb[:, :, 0]-mu, 2)+np.power(rgb[:, :, 1]-mu, 2)+np.power(rgb[:, :, 2]-mu, 2))/3)

#    pf.showImage('Saturation', S, 0)
    return S


def exposure(imgs):  # middle value is the best of pixel values (gaussian weighting)
    sig = 0.2
    n = len(imgs)
    h, w, c = imgs[0].shape[:3]
    E = np.zeros((h, w, n))
    for i in range(n):
        rgb = imgs[i]
        r = np.exp(-0.5*np.power(rgb[:, :, 0]-0.5, 2)/np.power(sig, 2))
        g = np.exp(-0.5*np.power(rgb[:, :, 1]-0.5, 2)/np.power(sig, 2))
        b = np.exp(-0.5*np.power(rgb[:, :, 2]-0.5, 2)/np.power(sig, 2))
        E[:, :, i] = r*g*b

#    pf.showImage('Exposure', E, 0)
    return E


def huekey(imgs, tHue, th):
    targHue = tHue/360
    halfRange = 180/360
    n = len(imgs)
    h, w, c = imgs[0].shape[:3]
    H = np.zeros((h, w, n))
    for i in range(n):
        hsv = cv2.cvtColor(np.uint8(imgs[i]*255), cv2.COLOR_BGR2HSV)
        hue = hsv[:, :, 0]/180
        e = abs(hue - targHue)
        e[e > halfRange] = 1 - e[e > halfRange]
        H[:, :, i] = 1 - 2*e
    H[H < th] = 0

#    pf.showImage('Hue', H, 0)
    return H


def saliency(imgs):
    n = len(imgs)
    h, w, c = imgs[0].shape[:3]
    S = np.zeros((h, w, n))
    for i in range(n):
        sm = pySaliencyMap.pySaliencyMap(w, h)
        S[:, :, i] = sm.SMGetSM(imgs[i])

#    pf.showImage('Saliency', S, 0)
    return S


def rgb2gray(img):
    return 0.2989*img[:, :, 0] + 0.5870*img[:, :, 1] + 0.1140*img[:, :, 2]


def matchingAve(imgs):
    n = len(imgs)
    meanImg = np.zeros((n, 3))
    minMean = 255
    imgMax = np.zeros((n, 3))
    for i in range(n):
        meanImg[i, :] = np.mean(imgs[i])
        if meanImg[i, 0] < minMean:
            minMean = meanImg[i, 0]
    for i in range(n):
        imgs[i] = minMean/meanImg[i, 0] * imgs[i]
        imgMax[i, :] = np.max(imgs[i])
    for i in range(n):
        imgs[i] /= np.min(imgMax[:, 0])


def demoFeatures():
    testFeature = 0

    if testFeature == 0:
        img = cv2.imread('img/flash/ambient.jpg')/255
        f = contrast([img])
        wname = 'contrast'
        imgNM = 'res/con_img.png'
        fNM = 'res/con_con.png'
    elif testFeature == 1:
        img = cv2.imread('img/magi/IMG_2616.jpg')/255
        f = saturation([img]) * 2
        wname = 'saturation'
        imgNM = 'res/sat_img.png'
        fNM = 'res/sat_sat.png'
    elif testFeature == 2:
        img = cv2.imread('img/house/B.jpg')/255
        f = exposure([img])
        wname = 'exposure'
        imgNM = 'res/exp_img.png'
        fNM = 'res/exp_exp.png'
    elif testFeature == 3:
        img = cv2.imread('img/huekey/DSC02746s.jpg')/255
        f = huekey([img], 60, 0.2)
        wname = 'huekey'
        imgNM = 'res/hue_img.png'
        fNM = 'res/hue_hue.png'
    elif testFeature == 4:
        img = cv2.imread('img/saliency/DSC02786s.bmp')/255
        f = saliency([img])
        wname = 'saliency'
        imgNM = 'res/sal_img.png'
        fNM = 'res/sal_sal.png'

    cv2.imshow('img', img)
    cv2.imshow(wname, f)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite(imgNM, np.uint8(255*img))
    cv2.imwrite(fNM, np.uint8(255*f*1.5))\


if __name__ == '__main__':
    demoFeatures()

