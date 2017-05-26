import cv2
import numpy as np
import copy

imgpath = 'D:\\DIP-Project1/b.jpg'
img = cv2.imread(imgpath)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('img', img)
row = len(img)
col = len(img[0])

def f(img, i, j, msize, maxormin, mr, mc):
    pxls = []
    for a in range(msize):
        for b in range(msize):
            mi = i + a - mr
            mj = j + b - mc
            pxls.append(img[mi][mj])
    pxls.sort()
    if maxormin == 0:
        return pxls[msize*msize-1]
    else:
        return pxls[0]

def filter(img, row, col, msize, maxormin):
    ret = copy.deepcopy(img)
    mr = (msize-1)//2
    mc = (msize-1)//2
    for i in range(mr, row-1-mr):
        for j in range(mc, col-1-mc):
            ret[i][j] = f(img, i, j, msize, maxormin, mr, mc)
    return ret

#maxormin=0,代表最大值过滤器; maxormin=1，代表最小值过滤器
d0 = 3
img1 = filter(img, row, col, d0, 0)
cv2.imshow('maxfilter', img1)
img2 = filter(img, row, col, d0, 1)
cv2.imshow('minfilter', img2)

cv2.waitKey(0)