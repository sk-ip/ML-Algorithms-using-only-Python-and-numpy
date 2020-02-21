import math

def calEntropy(dataset):
	numEntries = len(dataset)
	labelCounts = {}
	for featVec in dataset:
		currentLabel = featVec[-1]
		if currentLabel not in labelCounts.keys():
			labelCounts[currentLabel] = 0
		labelCounts[currentLabel] += 1
	
	entropy = 0.0

	for key in labelCounts:
		prob = float(labelCounts[key]) / numEntries
		entropy -= prob * math.log(prob, 2)
	
	return entropy


def splitDataSet(dataset, axis, value):
	retDataSet = []
	for featVec in dataset:
		if featVec[axis] == value:
			reducedFeatVec = featVec[:axis]
			reducedFeatVec.extends(featVec[axis+1:])
			retDataSet.append(reducedFeatVec)
	
	return retDataSet


def createTree(dataset, labels):
	