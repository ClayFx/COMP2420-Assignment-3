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
    # def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = np.shape(X)[0]
    clusterAssment = np.mat(np.zeros((m, 2)))  # create mat to assign data points
    # to a centroid, also holds SE of each point
    centers = randCent(X, n_cluster)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):  # for each data point assign it to the closest centroid
            minDist = np.inf
            minIndex = -1
            for j in range(n_cluster):
                distJI = distEclud(centers[j, :], X[i, :])
                if distJI < minDist:
                    minDist = distJI;
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist ** 2
        print(centers)
        for cent in range(n_cluster):  # recalculate centroids
            ptsInClust = X[np.nonzero(clusterAssment[:, 0].A == cent)[0]]  # get all the point in this cluster
            centers[cent, :] = np.mean(ptsInClust, axis=0)  # assign centroid to mean
    # return centroids, clusterAssment

    return centers, labels

def distEclud(vecA, vecB):
    return np.sqrt(sum(np.power(vecA - vecB, 2)))

def randCent(dataSet, k):
    n = np.shape(dataSet)[1]
    centroids = np.mat(np.zeros((k,n)))
    for j in range(n):
        minJ = min(dataSet[:,j])
        rangeJ = float(max(np.array(dataSet)[:,j]) - minJ)
        centroids[:,j] = minJ + rangeJ * np.random.rand(k,1)
    return centroids


## change the parameters of the function call to test your implementation
centers, labels = kmeans(X, n_cluster=4, random_seed=4, n_init=300)

print(centers,labels)