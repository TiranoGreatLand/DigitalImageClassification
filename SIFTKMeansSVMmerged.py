import numpy as np
import cv2


import readData

traindata, trainlabel, animalToInt, intToAnimal = readData.readTrainBWImage()
testdata, testlabel = readData.readTestBWImage(animalToInt)

print("data read over, feature extra and vocabulary make")

detector = cv2.xfeatures2d.SIFT_create()
wordCnt = 200

featureSet = np.float32([]).reshape(0, 128)

for im in traindata:
    kp, des = detector.detectAndCompute(im, None)
    featureSet = np.append(featureSet, des, axis=0)

featureCnt = featureSet.shape[0]

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.05)
flags = cv2.KMEANS_PP_CENTERS
compactness, labels, centers = cv2.kmeans(featureSet, wordCnt,
        bestLabels=None, criteria=criteria, attempts=100, flags=flags)

print("dictionary made over")

train_data = np.float32([]).reshape(0, wordCnt)
train_label = np.float32([])


tlen = len(traindata)


def featVecTransfer(features, centers):
    ret = np.zeros((1, wordCnt))
    for i in range(0, features.shape[0]):
        fi = features[i]
        diffMat = np.tile(fi, (wordCnt, 1)) - centers
        sqSum = (diffMat ** 2).sum(axis=1)
        dist = sqSum ** 0.5
        sortedIndices = dist.argsort()
        idx = sortedIndices[0]
        ret[0][idx] += 1
    return ret


for i in range(tlen):
    im = traindata[i]
    lb = trainlabel[i]
    kp, des = detector.detectAndCompute(im, None)
    vecf = featVecTransfer(des, centers)
    train_data = np.append(train_data, vecf, axis=0)

train_data = np.float32(train_data)

# train_label = np.float32(trainlabel)

print("train the SVM")

svm = cv2.ml.SVM_create()

svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setC(1.0)

train_label = np.array(trainlabel)

svm.train(train_data, cv2.ml.ROW_SAMPLE, train_label)

print("train over, predict")
tlen = len(testdata)

accuracy = 0

acs = {}

for i in range(tlen):
    im = testdata[i]
    lb = testlabel[i]
    if lb not in acs:
        acs[lb] = 0
    kp, des = detector.detectAndCompute(im, None)
    vecf = featVecTransfer(des, centers)
    case = np.float32(vecf)
    _, pred = svm.predict(case)
    pred = int(pred)
    print("predicted, ", pred, " true label ", lb)
    if pred == lb:
        accuracy += 1
        acs[lb] += 1

eachlen = tlen / len(acs)
for i in acs:
    print(intToAnimal[i]," prediction accuracy, ",  acs[i] / eachlen)
print(accuracy / tlen)
