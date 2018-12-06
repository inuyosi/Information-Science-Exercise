import cv2
import numpy as np
import matplotlib.pyplot as plt
import pyrFunc as pf
import imageFeatures as imf

global testMode
global tergetHue


def exposureFusion(imgs, param):
    try:
        testMode
    except NameError:
        testMode = 0
    try:
        tergetHue
    except NameError:
        tergetHue = 60

    print(param)
    conP = param[0]  # contrast parameter
    satP = param[1]  # saturation parameter
    expP = param[2]  # exposure parameter
    hueP = param[3]  # hue parameter
    salP = param[4]  # saliency parameter

    h, w, c = imgs[0].shape[:3]
    n = len(imgs)

    if testMode == 4:
        imf.matchingAve(imgs)

    W = np.ones((h, w, n))
    if conP > 0:
        print(' - contrast')
        W *= np.power(imf.contrast(imgs.copy()), conP)
    if satP > 0:
        print(' - saturation')
        W *= np.power(imf.saturation(imgs.copy()), satP)
    if expP > 0:
        print(' - exposure')
        W *= np.power(imf.exposure(imgs.copy()), expP)
    if hueP > 0:
        print(' - huekey')
        W *= np.power(imf.huekey(imgs.copy(), tergetHue, 0.2), hueP)
    if salP > 0:
        print(' - saliency')
        W *= np.power(imf.saliency(imgs.copy()), salP)

    W[W == 0] += 1e-12  # avoids division by zero
    Wsum = np.zeros((h, w))
    for i in range(n):
        Wsum += W[:, :, i]
    W /= np.tile(Wsum[:, :, None], [1, 1, n])  # normalization

#    showImage('W',W,0)

    # calc pyramid level
    m = int(np.log2(np.min((h, w)))) + 1

    # initialize pyr
    pyr = pf.generateGussianPyr(np.zeros((h, w, c)), m)

    # multi-scale processing
    for i in range(n):
        # generate Gaussian pyramid for Weight
        Gw = W[:, :, i].copy()
        gpW = pf.generateGussianPyr(Gw, m)

        # generate Laplacian pyramid for Image
        Gi = imgs[i].copy()
        lpI = pf.generateLaplacianPyr(Gi, m)

        # weighted average for pyramid
        for j in range(m):
            gpwtmp = gpW[j]
            w = np.tile(gpwtmp[:, :, None], [1, 1, c])
            pyr[j] += w*lpI[j]

    # reconstruct
    rImg = pf.reconstructLaplacianPyr(pyr.copy(), m)

    return rImg


if __name__ == '__main__':
    # testMode
    # [0] user settings
    # [1] standard demo
    # [2] flash non-flash demo
    # [3] omnifocal demo
    # [4] huekey demo
    # [5] saliency demo
    # [6] color demo
    # [7] demo for explanation
    # [8] standard 2 demo

    # select test mode
    testMode = 5

    # for [4] huekey demo
    # 0:red, 60:yellow, 120:green, 180:cyan, 240:blue, 320:magenta
    tergetHue = 60

    if testMode == 0:
        #######################
        # user settings
        #######################
        folderNM = './img/house/'
        fNM = ['A.jpg', 'B.jpg', 'C.jpg']
        con = 1.0  # contrast parameter
        sat = 1.0  # saturation parameter
        exp = 1.0  # exposure parameter
        hue = 0.0  # hue parameter
        sal = 1.0  # saliency parameter
    elif testMode == 1:
        #######################
        # standard demo
        #######################
        folderNM = './img/house/'
        fNM = ['A.jpg', 'B.jpg', 'C.jpg']
        con = 1.0  # contrast parameter
        sat = 1.0  # saturation parameter
        exp = 1.0  # exposure parameter
        hue = 0.0  # hue parameter
        sal = 0.0  # saliency parameter
    elif testMode == 2:
        #######################
        # flash non-flash demo
        #######################
        folderNM = './img/flash/'
        fNM = ['ambient.jpg', 'flash.jpg']
        con = 1.0  # contrast parameter
        sat = 1.0  # saturation parameter
        exp = 1.0  # exposure parameter
        hue = 0.0  # hue parameter
        sal = 0.0  # saliency parameter
    elif testMode == 3:
        #######################
        # omnifocal demo
        #######################
        folderNM = './img/magi/'
        fNM = ['IMG_2616.jpg', 'IMG_2617.jpg', 'IMG_2618.jpg']
        con = 1.0  # contrast parameter
        sat = 1.0  # saturation parameter
        exp = 0.0  # exposure parameter
        hue = 0.0  # hue parameter
        sal = 1.0  # saliency parameter
    elif testMode == 4:
        #######################
        # huekey demo
        #######################
        # 0:red, 60:yellow, 120:green, 180:cyan, 240:blue, 320:magenta
        #        tergetHue = 60;
        folderNM = './img/huekey/'
        fNM = ['DSC02748s.JPG', 'DSC02747s2.JPG', 'DSC02746s.JPG']
#        fNM = ['DSC02748s.JPG','DSC02747s.JPG','DSC02746s.JPG']
        con = 0.0  # contrast parameter
        sat = 0.0  # saturation parameter
        exp = 0.0  # exposure parameter
        hue = 1.0  # hue parameter
        sal = 0.0  # saliency parameter
    elif testMode == 5:
        #######################
        # saliency demo
        #######################
        folderNM = './img/saliency/'
        fNM = ['DSC02784s.bmp', 'DSC02785s.bmp', 'DSC02786s.bmp']
        con = 0.0  # contrast parameter
        sat = 0.0  # saturation parameter
        exp = 0.0  # exposure parameter
        hue = 0.0  # hue parameter
        sal = 1.0  # saliency parameter
    elif testMode == 6:
        #######################
        # color demo
        #######################
        folderNM = './img/rgb/'
        fNM = ['blue.png', 'green.png', 'red.png']
        con = 0.0  # contrast parameter
        sat = 0.0  # saturation parameter
        exp = 0.0  # exposure parameter
        hue = 1.0  # hue parameter
        sal = 0.0  # saliency parameter
    elif testMode == 7:
        #######################
        # demo for explanation
        #######################
        folderNM = './img/test/'
        fNM = ['im_contrasts.jpg', 'im_exposures.jpg', 'im_saturations.jpg']
        con = 1.0  # contrast parameter
        sat = 1.0  # saturation parameter
        exp = 1.0  # exposure parameter
        hue = 0.0  # hue parameter
        sal = 0.0  # saliency parameter
    elif testMode == 8:
        #######################
        # standard demo
        #######################
        folderNM = './img/sample/'
        fNM = ['high.JPG', 'mid.JPG', 'low.JPG']
        con = 1.0  # contrast parameter
        sat = 1.0  # saturation parameter
        exp = 1.0  # exposure parameter
        hue = 0.0  # hue parameter
        sal = 0.0  # saliency parameter

    imgs = [cv2.imread("{}{}".format(folderNM, fn))/255 for fn in fNM]
    exposureFusion(imgs, [con, sat, exp, hue, sal])
