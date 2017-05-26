import numpy as np


import readData

traindata, trainlabel, animalToInt, intToAnimal = readData.readTrainColorImage()
testdata, testlabel = readData.readTestColorImage(animalToInt)

print("data read over, translate")

train_data = []

test_data = []

for im in traindata:
    tmpl = np.zeros([768])
    row, col, deep = im.shape
    for i in range(row):
        for j in range(col):
            for k in range(deep):
                pxl = im[i][j][k]
                tmpl[256*k+pxl] += 1
    train_data.append(tmpl)

for im in testdata:
    tmpl = np.zeros([768])
    row, col, deep = im.shape
    for i in range(row):
        for j in range(col):
            for k in range(deep):
                pxl = im[i][j][k]
                tmpl[256*k+pxl] += 1
    test_data.append(tmpl)

print("data translate over, SVM train")

from sklearn.svm import SVC
from sklearn import metrics

clf = SVC(probability=False, kernel="rbf", C=2.8, gamma=0.01)

clf.fit(train_data, trainlabel)

print("train over, predict")

predicted = clf.predict(test_data)
print(metrics.confusion_matrix(testlabel, predicted))
print(metrics.accuracy_score(testlabel, predicted))
