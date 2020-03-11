import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# File import

dataSetFile = open("data/datingTestSet.txt")
noLines = len(dataSetFile.readlines())
dataSet = np.zeros((noLines, 3))
labels = []

dataSetFile = open("data/datingTestSet.txt")  # reset cursor
index = 0

for line in dataSetFile.readlines():
    line = line.strip()
    perLine = line.split("\t")
    dataSet[index, :] = perLine[0:3]
    labels.append(perLine[-1])
    index += 1

# Plotting

fig = plt.figure()
ax = plt.axes(projection='3d')
colorDict = {'largeDoses': 1, 'smallDoses': 2, 'didntLike': 3}
colors = []
for i in labels:
    colors.append(colorDict[i])
ax.scatter(dataSet[:, 0], dataSet[:, 1], dataSet[:, 2], c=np.array(colors))
plt.show()


# kNN algorithm

def classify(x, data, labels, k):
    dataSetSize = data.shape[0]
    distMat = ((((np.tile(x, (dataSetSize, 1)) - data) ** 2).sum(axis=1)) ** 0.5)
    '''
    diffMat = np.tile(x , (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis = 1)
    distances = sqDistances**0.5
    '''
    seqSort = distMat.argsort()
    kNeigbours = []
    labelCount = {}
    for i in range(k):
        kNeigbours.append(distMat[i])
        currLabel = labels[seqSort[i]]
        labelCount[currLabel] = labelCount.get(currLabel, 0) + 1
    return max(labelCount, key=labelCount.get)


# Normalization

minVals = dataSet.min(0)
maxVals = dataSet.max(0)
ranges = maxVals - minVals
normalData = np.zeros(np.shape(dataSet))
m = dataSet.shape[0]
normalData = (dataSet - np.tile(minVals, (m, 1))) / (np.tile(ranges, (m, 1)))


# Error calculation
# ratio = 0.10
# numTestVecs = int(m*ratio)
# error = 0.0
# for i in range(numTestVecs):
#     result = classify(normalData[i,:], normalData[numTestVecs:m,:],labels[numTestVecs:m],3)
#     print("The classifier came back with : ",result,", the real answer is: ", labels[i])
#     if(result != labels[i]):
#         error += 1.0
# print("The total error rate is ",error/float(numTestVecs)) 

def classifyPerson():
    resultList = {"didntLike": "not at all", "smallDoses": "in small doses", "largeDoses": "in large doses"}
    videoGames = float(input("How many hours does he spend playing video games(per year)? "))
    ffMiles = float(input("How many frequent flier miles does he have(per year)? "))
    iceCream = float(input("How many liters of ice cream does he consume per year?"))
    kNNresult = classify([videoGames, ffMiles, iceCream], dataSet, labels, 3)
    print("\nYou will probably like this person " + resultList[kNNresult])


classifyPerson()

