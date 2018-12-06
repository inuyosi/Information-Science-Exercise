import cv2
 
# read image file
img = cv2.imread('img/flash/ambient.jpg');
 
# some image processing ==============
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#   _____            _____
#  /   B/__         /   R/__
# /____/ G/__  ==> /____/ G/__
#   /____/ R/        /____/ B/
#     /____/           /____/
# ====================================
 
# write image file
cv2.imwrite('res/res.png', img)
 
# show image file
cv2.imshow('window name', img)
cv2.waitKey(0) # pause until any key pressed
cv2.destroyAllWindows() # close all windows
