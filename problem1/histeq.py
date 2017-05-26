import cv2
import numpy as np
import copy
import matplotlib.pyplot as plt

imgpath = 'D:\\DIP-Project1/a.jpg'
img = cv2.imread(imgpath)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow('img', img)

row = len(img)
col = len(img[0])

def histeq(img, row, col):
    ret = copy.deepcopy(img)
    act = np.zeros([256])
    for i in range(row):
        for j in range(col):
            pxl = img[i][j]
            act[pxl] += 1
    act /= (row*col)
    #显示每个灰度所包含的像素个数占总像素个数的百分比
    plt.plot(act)
    plt.show()
    nact = copy.deepcopy(act)
    count = 1
    while(count<256):
        nact[count] += nact[count-1]
        count += 1
    #显示转换过后每个灰度所包含的像素占总像素个数的百分比
    plt.plot(nact)
    plt.show()

    for i in range(row):
        for j in range(col):
            pxl = img[i][j]
            ret[i][j] = int(255*nact[pxl])
    return ret

img1 = histeq(img, row, col)
cv2.imshow('histeq', img1)

cv2.waitKey(0)