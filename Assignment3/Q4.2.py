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

    centers = randCent(X, n_cluster)  # 初始化质心，设置k=4
    labels = minDistance(X, centers)  # 第一次聚类迭代
    newVar = getVar(labels, centers)  # 获得均方误差值，通过新旧均方误差来获得迭代终止条件
    oldVar = -0.0001  # 旧均方误差值初始化为-1
    k = 1
    while abs(newVar - oldVar) >= 0.0001:  # 当计数器大于n_init时，迭代结束
        centers = getCentroids(labels)  # 获得新的质心
        labels = minDistance(X, centers)  # 新的聚类结果
        oldVar = newVar
        newVar = getVar(labels, centers)
        print('***** 第%d次迭代 *****' % k)
        k += 1
    return centers, labels


def minDistance(dataSet, centroidList):
    # 对每个属于dataSet的item，计算item与centroidList中k个质心的欧式距离，找出距离最小的，
    # 并将item加入相应的簇类中
    clusterDict = dict()  # 用dict来保存簇类结果
    for item in dataSet:
        vec1 = np.array(item)  # 转换成array形式
        flag = 0  # 簇分类标记，记录与相应簇距离最近的那个簇
        minDis = float("inf")  # 初始化为最大值

        for i in range(len(centroidList)):
            vec2 = np.array(centroidList[i])
            distance = calcuDistance(vec1, vec2)  # 计算相应的欧式距离
            if distance < minDis:
                minDis = distance
                flag = i  # 循环结束时，flag保存的是与当前item距离最近的那个簇标记

        if flag not in clusterDict.keys():  # 簇标记不存在，进行初始化
            clusterDict[flag] = list()
            # print flag, item
        clusterDict[flag].append(item)  # 加入相应的类别中
    return clusterDict  # 返回新的聚类结果


def getVar(clusterDict, centroidList):
    # 计算簇集合间的均方误差
    # 将簇类中各个向量与质心的距离进行累加求和
    sum = 0.0
    for key in clusterDict.keys():
        vec1 = np.array(centroidList[key])
        distance = 0.0
        for item in clusterDict[key]:
            vec2 = np.array(item)
            distance += calcuDistance(vec1, vec2)
        sum += distance
    return sum

def getCentroids(clusterDict):
    # 得到k个质心
    centroidList = list()
    for key in clusterDict.keys():
        centroid = np.mean(np.array(clusterDict[key]), axis=0)  # 计算每列的均值，即找到质心
        # print key, centroid
        centroidList.append(centroid)
    return np.array(centroidList).tolist()


def randCent(dataSet, k):
    # 初始化k个质心，随机获取
    n = np.shape(dataSet)[1]
    centroids = np.mat(np.zeros((k,n)))
    for j in range(n):
        minJ = min(dataSet[:,j])
        rangeJ = float(max(np.array(dataSet)[:,j]) - minJ)
        centroids[:,j] = minJ + rangeJ * np.random.rand(k,1)
    return centroids

def calcuDistance(vec1, vec2):
    # 计算向量vec1和向量vec2之间的欧氏距离
    return np.sqrt(np.sum(np.square(vec1 - vec2)))


def showCluster(centroidList, clusterDict):
    # 展示聚类结果
    colorMark = ['or', 'ob', 'og', 'ok', 'oy', 'ow']  # 不同簇类的标记 'or' --> 'o'代表圆，'r'代表red，'b':blue
    centroidMark = ['dr', 'db', 'dg', 'dk', 'dy', 'dw']  # 质心标记 同上'd'代表棱形
    for key in clusterDict.keys():
        plt.plot(centroidList[key][0], centroidList[key][1], centroidMark[key], markersize=12)  # 画质心点
        for item in clusterDict[key]:
            plt.plot(item[0], item[1], colorMark[key])  # 画簇类下的点
    plt.show()



## change the parameters of the function call to test your implementation
centers, labels = kmeans(X, n_cluster=4, random_seed=4, n_init=300)

print(centers, labels)

showCluster(centers, labels)  # 展示聚类结果