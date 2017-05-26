import cv2
import numpy as np

import readData

traindata, trainlabel, animalToInt, intToAnimal = readData.readTrainColorImage()
testdata, testlabel = readData.readTestColorImage(animalToInt)

trains = {}

tests = {}

tmplen = len(traindata)
for i in range(tmplen):
    im = traindata[i]
    lb = trainlabel[i]
    if lb not in trains:
        trains[lb] = []
    trains[lb].append(im)

tmplen = len(testdata)
for i in range(tmplen):
    im = testdata[i]
    lb = testlabel[i]
    if lb not in tests:
        tests[lb] = []
    tests[lb].append(im)


print("data read over, match start")

#detector = cv2.xfeatures2d.SIFT_create()
detector = cv2.xfeatures2d.SURF_create()

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params,search_params)

goodpercentage = 0.8

k = 10

def testmatchlists(im, trains):
    matchlists = []
    kp1, des1 = detector.detectAndCompute(im, None)
    #print(des1.shape)
    for i in trains:
        trims = trains[i]
        for tim in trims:
            kp2, des2 = detector.detectAndCompute(tim, None)
            matches = flann.knnMatch(des1, des2, k=2)
            count = 0
            for _, (m, n) in enumerate(matches):
                if m.distance < goodpercentage * n.distance:
            #        matchesMask[i] = [1, 0]
                    count += 1
            matchlists.append((count, i))
            #print(count, i)
    return matchlists

def knnGetter(k, matchlst):
    knnLabels = {}
    mlen = len(matchlst)
    used = np.zeros([mlen], int)
    for i in range(k):
        maxi = 0
        maxs, maxl = matchlst[0]
        for j in range(mlen):
            score, label = matchlst[j]
            if used[j] == 0 and score > maxs:
                maxs, maxl = matchlst[j]
                maxi = j
        used[maxi] = 1
        #print(maxl, maxs)
        if maxl not in knnLabels:
            knnLabels[maxl] = 0
        knnLabels[maxl] += 1
    maxl = 0
    maxt = 0
    for i in knnLabels:
        t = knnLabels[i]
        if t > maxt:
            maxt = t
            maxl = i
    return maxl

alltest = 0
accuracy = 0
for i in tests:
    teims = tests[i]
    tec = 0
    alltest += len(teims)
    ta = 0
    for im in teims:
        matchlists = testmatchlists(im, trains)
        judge = knnGetter(k, matchlists)
        #print(i, tec, " over")
        tec += 1
        if judge == i:
            accuracy += 1
            ta += 1
    print("case ", intToAnimal[i], " accuracy ", ta/len(teims))

print("total accuracy", accuracy/alltest)