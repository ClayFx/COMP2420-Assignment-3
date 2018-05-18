import json
import os
import urllib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from scipy.stats import ttest_ind, ttest_rel,ttest_1samp
from sklearn.preprocessing import scale


plt.style.use('seaborn-notebook')

# Q4.1

breast_cancer = datasets.load_breast_cancer()
X = scale(breast_cancer.data)

def kmeans(X, n_cluster, random_seed=2, n_init=100):
    '''
    Function calculates the centroids after performing k-means on the given dataset.
    Function returns two values new calculated centers and labels for each datapoint.
    If we have n_cluster = 4 then labels from algorithm will correspond to values 0,1,2 and 3

    Args:
        X: np.array representing set of input data
        n_cluster: number of clusters to use for clustering
        random_seed: random seed to use for calling random function in numpy
        n_inint: max number of iterations to use for k-means
    Returns:
        centers: np.array representing the centers for n_clusters
        labels: np.array containing a label for each datapoint in X
    '''
    centers = np.zeros((n_cluster, X.shape[1]))
    labels = np.zeros_like(X)
    # YOUR CODE HERE
    centers = randCent(X, n_cluster,random_seed)  # init
    labels = minDistance(X, centers)  # the first
    k = 1
    while k <= n_init:
        centers = getCentroids(labels)  # get the center
        labels = minDistance(X, centers)  # the new result
        print('***** the %dth iteration *****' % k)
        k += 1
    return centers, labels


def minDistance(dataSet, centroidList):
    # 对每个属于dataSet的item，计算item与centroidList中k个质心的欧式距离，找出距离最小的，
    # 并将item加入相应的簇类中
    clusterDict = dict()  # 用dict来保存簇类结果
    for item in dataSet:
        vec1 = np.array(item)
        flag = 0  # mark the nearest cluster
        minDis = float("inf")

        for i in range(len(centroidList)):
            vec2 = np.array(centroidList[i])
            distance = calcuDistance(vec1, vec2)
            if distance < minDis:
                minDis = distance
                flag = i  # record the nearest cluster

        if flag not in clusterDict.keys():  # the cluster flag does not exist, and do the initialization
            clusterDict[flag] = list()
            # print flag, item
        clusterDict[flag].append(item)  # add it into the corresponding cluster
    return clusterDict


def getCentroids(clusterDict):
    # get k center
    centroidList = list()
    for key in clusterDict.keys():
        centroid = np.mean(np.array(clusterDict[key]), axis=0)  # calculate the mean to get the center
        # print key, centroid
        centroidList.append(centroid)
    return np.array(centroidList).tolist()


def randCent(dataSet, k, random_seed):
    # init k center
    n = np.shape(dataSet)[1]
    np.random.seed(random_seed)
    centroids = np.mat(np.zeros((k,n)))
    for j in range(n):
        minJ = min(dataSet[:,j])
        rangeJ = float(max(np.array(dataSet)[:,j]) - minJ)
        centroids[:,j] = minJ + rangeJ * np.random.rand(k,1)
    return centroids

def calcuDistance(vec1, vec2):
    # calculate the distance of two vectors
    return np.sqrt(np.sum(np.square(vec1 - vec2)))


def showCluster(centroidList, clusterDict):
    # show the result
    colorMark = ['or', 'ob', 'og', 'ok', 'oy', 'ow']
    centroidMark = ['dr', 'db', 'dg', 'dk', 'dy', 'dw']
    for key in clusterDict.keys():
        plt.plot(centroidList[key][0], centroidList[key][1], centroidMark[key], markersize=12)
        for item in clusterDict[key]:
            plt.plot(item[0], item[1], colorMark[key])
    plt.show()



## change the parameters of the function call to test your implementation
centers, labels = kmeans(X, n_cluster=2, random_seed=2, n_init=300)

print(centers, labels)

showCluster(centers, labels)