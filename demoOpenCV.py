import cv2
img = cv2.imread('img/flash/ambient.jpg') # read image file
bimg = cv2.GaussianBlur(img, (51,51), 5) # gaussian filter (size=(51,51),sigma=5)
cv2.imshow('img',img)
cv2.imshow('blur img',bimg)
cv2.waitKey(0) # pause until press any key
cv2.destroyAllWindows # close all cv2's windows
