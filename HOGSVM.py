import numpy as np
import cv2


import readData

traindata, trainlabel, animalToInt, intToAnimal = readData.readTrainBWImage()
testdata, testlabel = readData.readTestBWImage(animalToInt)

print("data read over, hog extract")

hog = cv2.HOGDescriptor()

train_data = []
test_data = []
for im in traindata:
    desc = hog.compute(im)
    #print(desc.shape)
    train_data.append(desc)
for im in testdata:
    desc = hog.compute(im)
    test_data.append(desc)
train_data = np.array(train_data)
test_data = np.array(test_data)
trainlabel = np.array(trainlabel)

svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setC(1.0)

print("train SVM")

svm.train(train_data, cv2.ml.ROW_SAMPLE, trainlabel)
(ret, res) = svm.predict(test_data)

llen = len(testlabel)

acc = {}

for i in range(llen):
    lb = testlabel[i]
    pred = int(res[i][0])
    if lb not in acc:
        acc[lb] = 0
    if lb == pred:
        acc[lb] += 1

sl = llen/len(acc)

sumacc = 0

for i in acc:
    print(intToAnimal[i], " accuracy, ", acc[i]/sl)
    sumacc += acc[i]
print("all: ", sumacc/llen)

print("svm from sklearn")

testlabel = np.array(testlabel)

from sklearn.svm import SVC
from sklearn import metrics

clf = SVC(probability=False, kernel="rbf", C=2.8, gamma=0.01)

tr = []
te = []
for i in train_data:
    ilen = len(i)
    ti = np.zeros([ilen])
    for j in range(ilen):
        ti[j] = i[j][0]
    tr.append(ti)
for i in test_data:
    ilen = len(i)
    ti = np.zeros([ilen])
    for j in range(ilen):
        ti[j] = i[j][0]
    te.append(ti)

print("start *******************************************")
clf.fit(tr, trainlabel)

predicted = clf.predict(te)
print(metrics.confusion_matrix(testlabel, predicted))
print(metrics.accuracy_score(testlabel, predicted))