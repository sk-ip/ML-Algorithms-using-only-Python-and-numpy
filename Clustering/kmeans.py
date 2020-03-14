from numpy import *
import random
import math


def loadDataset():
    return [10, 11, 12, 30, 31, 32, 87, 86, 86, 86, 85]


def calculateMeans(dataset):
    return sum(dataset) / len(dataset)


def getRandomMeans(k, dataset):
    return list(random.sample(set(dataset), k))


def createClusters(k, means, dataset):
    clusters = [[] for i in range(k)]

    if means:
        pass
    else:
        means = getRandomMeans(k, dataset)

    for num in dataset:
        minDistances = map(lambda mean, num: abs(mean - num), means, [num] * k)
        minDistances = list(minDistances)
        index = minDistances.index(min(minDistances))
        clusters[index].append(num)

    return clusters


def kmeans(k, dataset):
    if isinstance(k, float):
        print("k should be integer value")
        return None

    new_means = random.sample(range(1, 10), k)
    prev_means = random.sample(range(2, 20), k)

    while new_means != prev_means:
        prev_means = new_means.copy()
        clusters = createClusters(k, None, dataset)
        new_means = list(map(int, map(calculateMeans, clusters)))
        print(new_means)
        print(clusters)


if __name__ == "__main__":
    data = loadDataset()
    kmeans(3, data)  # the value of k is 3 here
