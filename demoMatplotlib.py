import matplotlib.pyplot as plt
img = plt.imread('img/flash/ambient.jpg')
plt.subplot(121),plt.imshow(img)
plt.subplot(122),plt.imshow(255-img) # negative-positive conversion
plt.show() # required for showing figure