import cv2
import numpy as np
import copy
from matplotlib import pyplot as plt

#读取图片
imgpath = 'D:\\Codes\\demoofDIP/aaa.tif'
img = cv2.imread(imgpath, 0)
cv2.imshow('img', img)
#傅里叶变换
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
#高斯高通滤波器
row, col = fshift.shape
n = 2
d0 = 80
n1 = row//2
n2 = col//2
for i in range(row):
    for j in range(col):
        d = np.sqrt((i - n1) * (i - n1) + (j - n2) * (j - n2))
        h = 1 - np.exp(-d*d/2/(d0*d0))
        fshift[i][j] = fshift[i][j] * h
#逆傅里叶变换
result = np.fft.ifftshift(fshift)
result = np.fft.ifft2(result)
result = np.uint8(np.abs(result))
#显示图片
plt.imshow(result, cmap='gray')
plt.show()

cv2.waitKey(0)