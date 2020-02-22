import math
import pandas as pd
import csv
import operator
import numpy as np

def loadDataset():
	with open("baseball.csv", newline="\n") as f:
		reader = csv.reader(f)
		dataset = list(reader)
	return dataset


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
			reducedFeatVec.extend(featVec[axis+1:])
			retDataSet.append(reducedFeatVec)
	
	return retDataSet


def chooseBestFeatureToSplit(dataset):
	numFeatures = len(dataset[0])-1
	baseEntropy = calEntropy(dataset)
	bestInfoGain = 0.0; bestFeature = -1
	for i in range(numFeatures):
		featList = [example[i] for example in dataset]
		uniqueVals = set(featList)
		newEntropy = 0.0
		for value in uniqueVals:
			subDataset = splitDataSet(dataset, i, value)
			prob = len(subDataset) / float(len(dataset))
			newEntropy += prob * calEntropy(subDataset)
		infoGain = baseEntropy - newEntropy
		if infoGain > bestInfoGain:
			bestInfoGain = infoGain
			bestFeature = i
	return bestFeature


def majorityCount(classList):
	classCounts = {}
	for vote in classList:
		if vote not in classCounts.keys():
			classCounts[vote] = 0
		classCounts[vote] += 1
	sortedClassCount = sorted(classCounts.items(), key = operator.itemgetter(1))
	return sortedClassCount[0][0]


def createTree(dataset, labels):
	classList = [example[-1] for example in dataset]
	if classList.count(classList[0]) == len(classList):
		return classList[0]
	if len(dataset[0]) == 1:
		return majorityCount(classList)
	
	bestFeature = chooseBestFeatureToSplit(dataset)
	bestFeatureLabel = labels[bestFeature]

	tree = {bestFeatureLabel:{}}
	del labels[bestFeature]

	featValues = [example[bestFeature] for example in dataset]
	uniqueVals = set(featValues)
	for value in uniqueVals:
		subLables = labels[:]
		tree[bestFeatureLabel][value] = createTree(splitDataSet(dataset, bestFeature, value), subLables)
	
	return tree


if __name__ == "__main__":
	dataset = np.array(loadDataset())
	dataset = np.delete(dataset, (0), axis=0)
	labels = dataset[:,-1]
	dataset = np.delete(dataset,(4), axis=1)
	tree = createTree(dataset.tolist(), labels.tolist())
	print(tree)

