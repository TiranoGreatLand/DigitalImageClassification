import cv2
import numpy as np
import copy

imgpath = 'D:\\DIP-Project1/b.jpg'
img = cv2.imread(imgpath, 0)
#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('img', img)
row = len(img)
col = len(img[0])

def autoadp(img, i, j, row, col):
    lb = 1
    if row>col:
        mlb = row//2
    else:
        mlb = col//2
    while lb<=mlb:
        if i-lb<0 or j-lb<0 or i+lb>=row or j+lb>=col:
            return img[i][j]
        else:
            msize = 2*lb+1
            pxls = []
            for a in range(msize):
                for b in range(msize):
                    pxls.append(img[i-lb+a][j-lb+b])
            median = np.median(pxls)
            min = np.min(pxls)
            max = np.max(pxls)
            if median>min and median<max:
                if img[i][j]>min and img[i][j]<max:
                    return img[i][j]
                else:
                    return median
        lb += 1
    return img[i][j]

def adaptivefilter(img, row, col):
    ret = copy.deepcopy(img)
    for i in range(row):
        for j in range(col):
            ret[i][j] = autoadp(img, i, j, row, col)
    return ret

rimg = adaptivefilter(img, row, col)
cv2.imshow('adpimg', rimg)

rimg = adaptivefilter(rimg, row, col)
cv2.imshow('adp2img', rimg)

rimg = adaptivefilter(rimg, row, col)
cv2.imshow('adp3img', rimg)

rimg = adaptivefilter(rimg, row, col)
cv2.imshow('adp4img', rimg)

cv2.waitKey(0)