import cv2
import numpy as np
import copy

imgpath = 'D:\\DIP-Project1/b.jpg'
img = cv2.imread(imgpath)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('img', img)
row = len(img)
col = len(img[0])

def ma(img, i, j, msize, mr, mc):
    sum = 0

    for a in range(msize):
        for b in range(msize):
            mi = i + a - mr
            mj = j + b - mc
            sum += img[mi][mj]

    return int(sum/(msize*msize))

def avg(img, row, col, msize=3):
    rimg = copy.deepcopy(img)

    mr = (msize-1)//2
    mc = (msize-1)//2
    for i in range(mr, row-mr-1):
        for j in range(mc, col-mc-1):
            rimg[i][j] = ma(img, i, j, msize, mr, mc)

    return rimg

d0 = 9

rimg = avg(img, row, col, d0)
cv2.imshow('aimg', rimg)
cv2.waitKey(0)