from numpy import *
import operator


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]  # shape[0] stands for the num of row
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet  # Subtract element-wise
    sqDiffMat = diffMat ** 2  # squared for the subtract
    sqDistances = sqDiffMat.sum(axis=1)  # sum is performed by row
    distance = sqDistances ** 0.5

    sortedDistIndices = argsort(distance)

    classCount = {}  # define a dictionary (can be append element)
    for i in xrange(k):
        # # step 3: choose the min k distance
        voteLabel = labels[sortedDistIndices[i]]

        ## step 4: count the times labels occur
        # when the key voteLabel is not in dictionary classCount, get()
        # will return 0
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)

    return sortedClassCount[0][0]


# change a filename string and output two things: a matrix of training
# examples and a vector of class labels.
def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())     # counts the number of lines
    returnMat = zeros((numberOfLines, 3))   # create Numpy N*[0 0 0] matrix to return
    classLabelVector = []
    #enumLabelVector = []
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]     #[0:3] means 0,1,2
        classLabelVector.append(int(listFromLine[-1]))
        # if cmp(listFromLine[-1],'didntLike')==0:
        #     enumLabelVector.append(1)
        # if cmp(listFromLine[-1],'smallDoses')==0:
        #     enumLabelVector.append(2)
        # if cmp(listFromLine[-1],'largerDoses')==0:
        #     enumLabelVector.append(3)
        index += 1
    fr.close()
    return returnMat, classLabelVector #enumLabelVector
