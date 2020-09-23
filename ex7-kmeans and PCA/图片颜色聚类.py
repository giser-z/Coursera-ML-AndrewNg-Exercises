"""
案例2: 使用kmeans对图片颜色进行聚类
RGB图像，每个像素点值范围0-255
数据集：data/bird_small.mat，data/bird_small.png
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from skimage import io


def find_centroids(X, centros):
    idx = []
    for i in range(len(X)):
        # (2,)(k,2)->(k,2)
        dist = np.linalg.norm((X[i] - centros), axis=1)  # (k,)
        id_i = np.argmin(dist)
        idx.append(id_i)
    return np.array(idx)


def compute_centros(X, idx, k):
    centros = []

    for i in range(k):
        centros_i = np.mean(X[idx == i], axis=0)
        centros.append(centros_i)

    return np.array(centros)


def run_kmeans(X, centros, iters):
    k = len(centros)
    centros_all = []
    centros_all.append(centros)
    centros_i = centros
    for i in range(iters):
        idx = find_centroids(X, centros_i)
        centros_i = compute_centros(X, idx, k)
        centros_all.append(centros_i)

    return idx, np.array(centros_all)


def init_centros(X, k):
    index = np.random.choice(len(X), k)
    return X[index]


if __name__ == '__main__':

    data = sio.loadmat('data/bird_small.mat')
    A = data['A']
    print(A.shape)

    image = io.imread('data/bird_small.png')
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.imshow(image)
    plt.axis('off') # 不显示坐标轴


    A = A / 255
    A = A.reshape(-1, 3) # 三维->二维
    print(A.shape)
    k = 16
    idx, centros_all = run_kmeans(A, init_centros(A, k=16), iters=20)
    centros = centros_all[-1] # 迭代最后一次的聚类中心点坐标
    im = np.zeros(A.shape)
    for i in range(k):
        im[idx == i] = centros[i]
    im = im.reshape(128, 128, 3)
    # im = np.rint(im*255).astype('uint8') # 还原为0-255的rgb范围图像
    print(im)
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.imshow(im)
    plt.show()
