import os
import cv2

#datapath = "D:\DataGames\dipdata4"
datapath = "D:\DataGames\dipdata6"

def readTrainColorImage():
    traindata = []
    trainlabel = []
    count = 0
    animalToInt = {}
    intToAnimal = {}
    trainpath = os.path.join(datapath, "train")
    for animal in os.listdir(trainpath):
        if animal not in animalToInt:
            animalToInt[animal] = count
            intToAnimal[count] = animal
        subpath = os.path.join(trainpath, animal)
        print(subpath, " data read")
        for each in os.listdir(subpath):
            tmppath = os.path.join(subpath, each)
            im = cv2.imread(tmppath)
            im = cv2.resize(im, (246, 228))
            traindata.append(im)
            trainlabel.append(count)
        count += 1
    print(" color train data read over")
    return traindata, trainlabel, animalToInt, intToAnimal

def readTestColorImage(animalToInt):
    testdata = []
    testlabel = []
    testpath = os.path.join(datapath, "test")
    for animal in os.listdir(testpath):
        label = animalToInt[animal]
        subpath = os.path.join(testpath, animal)
        print(subpath, " data read")
        for each in os.listdir(subpath):
            tmppath = os.path.join(subpath, each)
            im = cv2.imread(tmppath)
            im = cv2.resize(im, (246, 228))
            testdata.append(im)
            testlabel.append(label)
    print(" color test data read over")
    return testdata, testlabel

def readTrainBWImage():
    traindata = []
    trainlabel = []
    count = 0
    animalToInt = {}
    intToAnimal = {}
    trainpath = os.path.join(datapath, "train")
    for animal in os.listdir(trainpath):
        if animal not in animalToInt:
            animalToInt[animal] = count
            intToAnimal[count] = animal
        subpath = os.path.join(trainpath, animal)
        print(subpath, " data read")
        for each in os.listdir(subpath):
            tmppath = os.path.join(subpath, each)
            im = cv2.imread(tmppath, 0)
            im = cv2.resize(im, (246, 228))
            traindata.append(im)
            trainlabel.append(count)
        count += 1
    print(" BW train data read over")
    return traindata, trainlabel, animalToInt, intToAnimal

def readTestBWImage(animalToInt):
    testdata = []
    testlabel = []
    testpath = os.path.join(datapath, "test")
    for animal in os.listdir(testpath):
        label = animalToInt[animal]
        subpath = os.path.join(testpath, animal)
        print(subpath, " data read")
        for each in os.listdir(subpath):
            tmppath = os.path.join(subpath, each)
            im = cv2.imread(tmppath, 0)
            im = cv2.resize(im, (246, 228))
            testdata.append(im)
            testlabel.append(label)
    print(" BW test data read over")
    return testdata, testlabel

