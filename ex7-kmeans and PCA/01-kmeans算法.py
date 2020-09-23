"""
案例1: 给定一个二维数据集，使用kmeans进行聚类
数据集：data/ex7data2.mat
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from skimage import io


def find_centroids(X, centros): # 获取每个样本所属的类别
    idx = []
    for i in range(len(X)):
        # (2,)(k,2)->(k,2)
        dist = np.linalg.norm((X[i] - centros), axis=1)  # (k,) 每一个样本点到中心点的距离，axis=1按行向量处理.dist->(3,)
        id_i = np.argmin(dist)  # 保存样本点到三个中心点最小距离的索引值。--索引值共有0,1,2三个，即三类
        idx.append(id_i)
    return np.array(idx)  # 每一个样本点对应一个类别(k)，idx是按样本点顺序对应的分类标签数组


def compute_centros(X, idx, k): # 计算聚类中心点
    centros = []

    for i in range(k):
        centros_i = np.mean(X[idx == i], axis=0) # 计算每类中心点坐标(x,y)--axis=0压缩行，对每列求平均
        centros.append(centros_i)

    return np.array(centros)


def run_kmeans(X, centros, iters):
    k = len(centros)
    centros_all = [centros]  # 用来存储每一次迭代更新的聚类中心点坐标(X,y)
    centros_i = centros
    for i in range(iters):
        idx = find_centroids(X, centros_i)
        centros_i = compute_centros(X, idx, k)
        centros_all.append(centros_i)

    return idx, np.array(centros_all)  # idx返回最后一次迭代更新的分类结果，centros_all聚类迭代过程中所有的中心点坐标


def plot_data(X, centros_all, idx): # 绘制数据集和聚类中心的移动轨迹
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=idx, cmap='rainbow')
    plt.plot(centros_all[:, :, 0], centros_all[:, :, 1], 'kx--')


def init_centros(X,k):  # 随机初始化k个聚类中心点
    index = np.random.choice(len(X),k)
    return X[index]


if __name__ == '__main__':

    data1 = sio.loadmat('data/ex7data2.mat')
    print(data1)
    X = data1['X']
    plt.scatter(X[:, 0], X[:, 1])
    # 人为初始化三个聚类中心点
    centros = np.array([[3, 3], [6, 2], [8, 5]]) # 初始化三个聚类中心点(在此认为选择了三个点)，即分三类
    idx = find_centroids(X, centros)
    idx, centros_all = run_kmeans(X, centros, iters=10)
    plot_data(X, centros_all, idx)

    # 随机初始化三个聚类中心点
    for i in range(4):
        idx, centros_all = run_kmeans(X, init_centros(X, k=3), iters=10)
        plot_data(X, centros_all, idx)

    plt.show()
