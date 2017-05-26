import numpy as np


import readData

traindata, trainlabel, animalToInt, intToAnimal = readData.readTrainBWImage()
testdata, testlabel = readData.readTestBWImage(animalToInt)

print("data read over, translate")

train_data = []

test_data = []

for im in traindata:
    iml = np.zeros([246 * 228])
    for i in range(228):
        for j in range(246):
            iml[246 * i + j] = im[i][j]
    train_data.append(iml)

for im in testdata:
    iml = np.zeros([246 * 228])
    for i in range(228):
        for j in range(246):
            iml[246 * i + j] = im[i][j]
    test_data.append(iml)

print("data translate over, SVM train")

from sklearn.svm import SVC
from sklearn import metrics

clf = SVC(probability=False, kernel="rbf", C=2.8, gamma=0.01)

clf.fit(train_data, trainlabel)

print("train over, predict")

predicted = clf.predict(test_data)
print(metrics.confusion_matrix(testlabel, predicted))
print(metrics.accuracy_score(testlabel, predicted))
