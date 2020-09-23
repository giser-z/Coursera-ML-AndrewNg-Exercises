"""
案例1：使用PCA进行二维数据的降维
数据集：data/ex7data1.mat
"""

import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context="notebook", style="white")

mat = sio.loadmat('./data/ex7data1.mat')
X = mat['X']  # X-->(50,2)
print(X.shape)
# 散点图
plt.figure()
plt.scatter(X[:,0],X[:,1])

# 对X去均值化
X_demean = X - np.mean(X, axis=0)
plt.figure()
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
plt.title('X去均值化的散点图')
plt.scatter(X_demean[:,0],X_demean[:,1])


C = X_demean.T @ X_demean / len(X) # 计算协方差矩阵n×n，n表示特征数量
print(C)
U, S, V = np.linalg.svd(C) # 计算特征值，特征向量。SVD奇异值分解

U_reduce = U[:,0]  # 选取前K向量，获得n×k的矩阵。此处选U的第一列(2,)
z = X_demean @ U_reduce  # 新特征向量z-->(50,1)

plt.figure(figsize=(5,5))
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
plt.title('数据降维')
plt.scatter(X_demean[:,0], X_demean[:,1])
plt.plot([0,U_reduce[0]], [0,U_reduce[1]], c='r')
plt.plot([0,U[:,1][0]], [0,U[:,1][1]], c='k')

# 还原数据---z为前面PCA压缩的数据，还原回原始的二维空间
X_approx = z.reshape(50,1) @ U_reduce.reshape(1,2) + np.mean(X,axis=0) # X_approx-->(50,2)近似等于原X

plt.figure()
plt.scatter(X[:,0], X[:,1])
plt.scatter(X_approx[:,0], X_approx[:,1])



