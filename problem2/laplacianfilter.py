import cv2
import numpy as np
import copy

imgpath = 'D:\\DIP-Project1/b.jpg'
img = cv2.imread(imgpath)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('img', img)
row = len(img)
col = len(img[0])

def pxlcom(img, i, j):
    ret = int(img[i-1][j])+img[i+1][j]+img[i][j-1]+img[i][j+1]-4*img[i][j]
    if ret>255:
        ret = 255
    elif ret<0:
        ret = 0
    return ret

def laplacianfilter(img, row, col):
    ret = copy.copy(img)
    for i in range(1, row-1):
        for j in range(1, col-1):
            ret[i][j] = pxlcom(img, i, j)
    return ret

def imgsub(img, rimg, row, col):
    ret = copy.copy(img)
    for i in range(1, row-1):
        for j in range(1, col-1):
            if rimg[i][j]>ret[i][j]:
                ret[i][j] = 0
            else:
                ret[i][j] -= rimg[i][j]
    return ret

rimg = laplacianfilter(img, row, col)

cv2.imshow("laplacianfilter", rimg)

limg = imgsub(img, rimg, row, col)

cv2.imshow("result", limg)

cv2.waitKey(0)