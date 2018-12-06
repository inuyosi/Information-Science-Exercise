#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np
import exposureFusion as ef


def value_changed(*args):
    update_resImg()


def update_resImg():
    global imgs
    param = [scale1.get(), scale2.get(), scale3.get(), 0, scale4.get()]
    rImg = getFusedImage(imgs, param)
    label4.configure(image=rImg, width=w2, height=h2)
    label4.image = rImg


def update_inputImg(folderNM, fNM):
    n = len(fNM)
    fNM1 = fNM[0]
    fNM2 = fNM[1]
    if n == 2:
        fNM3 = fNM[1]
    else:
        fNM3 = fNM[2]

    # label1: img1
    img1 = ImageTk.PhotoImage(Image.open(
        (folderNM+fNM1)).resize((w1, h1), Image.ANTIALIAS))
    label1.configure(image=img1, width=w1, height=h1)
    label1.image = img1

    # label2: img2
    img2 = ImageTk.PhotoImage(Image.open(
        (folderNM+fNM2)).resize((w1, h1), Image.ANTIALIAS))
    label2.configure(image=img2, width=w1, height=h1)
    label2.image = img2

    # label3: img3
    img3 = ImageTk.PhotoImage(Image.open(
        (folderNM+fNM3)).resize((w1, h1), Image.ANTIALIAS))
    label3.configure(image=img3, width=w1, height=h1)
    label3.image = img3


def getFusedImage(imgs, param):
    if 0:
        # Exposure Fusion using OpenCV
        ims = imgs*255
        merge_mertens = cv2.createMergeMertens(param[0], param[1], param[2])
        tmp = merge_mertens.process(ims)
        tmp = (tmp-tmp.min())/(tmp.max()-tmp.min())
    else:
        tmp = ef.exposureFusion(imgs, param)
    tmp[tmp < 0] = 0
    tmp[tmp > 1] = 1
    rImg = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(
        np.uint8(tmp*255), cv2.COLOR_BGR2RGB)).resize((w2, h2), Image.ANTIALIAS))
    return rImg


def calcSize(folderNM, fNM):
    tmp = Image.open((folderNM+fNM[0]))
    imgw, imgh = tmp.size
    h1 = 125
    w1 = int(imgw * h1 / imgh)
    h2 = 400
    w2 = int(imgw * h2 / imgh)
    return h1, w1, h2, w2


def button_click(*args):
    global imgs
    dNM = text1.get('1.0', 'end -1c')
    folderNM = './img/'+dNM+'/'
    fNM = getfNM(folderNM)
    imgs = [cv2.imread("{}{}".format(folderNM, fn))/255 for fn in fNM]

    n = len(fNM)
    if n <= 1:
        return
    elif n > 3:
        print('we show only 3 input images. (3 of '+str(n)+')')

    changeTarget(folderNM, fNM, imgs)


def getfNM(folderNM):
    fNM = []
    fnms = os.listdir(folderNM)
    for fnm in fnms:
        root, ext = os.path.splitext(fnm)
        if ext == '.png' or ext == '.jpg' or ext == '.bmp' or ext == '.JPG':
            fNM.append(fnm)
    return fNM


def changeTarget(folderNM, fNM, imgs):
    global h1, w1, h2, w2
    h1, w1, h2, w2 = calcSize(folderNM, fNM)
    update_inputImg(folderNM, fNM)
    update_resImg()


if __name__ == '__main__':
    folderNM = './img/sample1/'
    fNM = getfNM(folderNM)
    imgs = [cv2.imread("{}{}".format(folderNM, fn))/255 for fn in fNM]
    param = [1, 1, 1, 0, 0]

    # set root of window
    root = tk.Tk()
    root.title('Demo: Exposure Fusion')
    root.geometry('1280x720')
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    # set frame
    frame0 = tk.Frame(root)
    frame0.grid(row=0, column=0, sticky=tk.NSEW)

    frame1 = tk.Frame(root)
    frame1.grid(row=1, column=0, sticky=tk.NSEW)

    frame2 = tk.Frame(root)
    frame2.grid(row=1, column=1, sticky=tk.NSEW)

    frame3 = tk.Frame(root)
    frame3.grid(row=2, column=0, columnspan=2, sticky=tk.NSEW)

    ###################################################################
    ## frame0 #########################################################
    ###################################################################
    # label0
    label0 = tk.Label(frame0, anchor='w', text='Directory name: ')
    label0.grid(row=0, column=0, sticky=tk.NSEW)

    # text1
    text1 = tk.Text(frame0, width=10, height=1)
    text1.grid(row=0, column=1, sticky=tk.NSEW)

    # button1
    button1 = tk.Button(
        frame0,
        text='OK',
        command=button_click)
    button1.grid(row=0, column=2, padx=5)
    # ==================================================================

    h1, w1, h2, w2 = calcSize(folderNM, fNM)

    ###################################################################
    ## frame1 #########################################################
    ###################################################################
    # label1: img1
    img1 = ImageTk.PhotoImage(Image.open(
        (folderNM+fNM[0])).resize((w1, h1), Image.ANTIALIAS))
    label1 = tk.Label(frame1, image=img1, width=w1, height=h1)
    label1.pack(fill='both', expand='y')

    # label2: img2
    img2 = ImageTk.PhotoImage(Image.open(
        (folderNM+fNM[1])).resize((w1, h1), Image.ANTIALIAS))
    label2 = tk.Label(frame1, image=img2, width=w1, height=h1)
    label2.pack(fill='both', expand='y')

    # label3: img3
    img3 = ImageTk.PhotoImage(Image.open(
        (folderNM+fNM[2])).resize((w1, h1), Image.ANTIALIAS))
    label3 = tk.Label(frame1, image=img3, width=w1, height=h1)
    label3.pack(fill='both', expand='y')
    # ==================================================================

    ###################################################################
    ## frame2 #########################################################
    ###################################################################
    # label4: output img
#    img4 = ImageTk.PhotoImage(.resize((w2,h2),Image.ANTIALIAS))
    img4 = getFusedImage(imgs, param)
    label4 = tk.Label(frame2, image=img4, width=w2, height=h2)
    label4.pack(fill='both', expand='y')
    # ==================================================================

    ###################################################################
    ## frame3 #########################################################
    ###################################################################
    # scale1 (contrast)
    scale1 = tk.Scale(
        frame3,
        label='contrast param',
        orient='h',
        length=200,
        from_=0.0,
        to=2.0,
        resolution=0.1,
        command=value_changed)
    scale1.pack(fill='both', expand='y')
    scale1.set(1.0)

    # scale2 (saturation)
    scale2 = tk.Scale(
        frame3,
        label='saturation param',
        orient='h',
        length=200,
        from_=0.0,
        to=2.0,
        resolution=0.1,
        command=value_changed)
    scale2.pack(fill='both', expand='y')
    scale2.set(1.0)

    # scale3 (exposure)
    scale3 = tk.Scale(
        frame3,
        label='exposure param',
        orient='h',
        length=200,
        from_=0.0,
        to=2.0,
        resolution=0.1,
        command=value_changed)
    scale3.pack(fill='both', expand='y')
    scale3.set(1.0)

    # scale4 (saliency)
    scale4 = tk.Scale(
        frame3,
        label='saliency param',
        orient='h',
        length=200,
        from_=0.0,
        to=2.0,
        resolution=0.1,
        command=value_changed)
    scale4.pack(fill='both', expand='y')
    scale4.set(0.0)
    # ==================================================================

    root.mainloop()
