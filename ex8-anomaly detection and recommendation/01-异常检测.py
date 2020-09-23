"""
案例1: 检测异常服务器
数据集：data/ex8data1.mat
注：算法手动实现
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt


def estimate_gaussian(X, isCovariance):  # 计算均值与协方差矩阵
    means = np.mean(X, axis=0)
    if isCovariance:
        sigma2 = (X - means).T @ (X - means) / len(X)  # 主对角线上的数值是方差，其他是协方差，也可用sigma=np.cov(X.T)
    else:
        sigma2 = np.var(X, axis=0)  # np.var计算方差
    return means, sigma2


def gaussian_distribution(X, means, sigma2):
    if np.ndim(sigma2) == 1:  # 数组维度是2，为原高斯分布模型
        sigma2 = np.diag(sigma2)

    X = X - means
    n = X.shape[1]

    first = np.power(2 * np.pi, -n / 2) * (np.linalg.det(sigma2) ** (-0.5))
    second = np.diag(X @ np.linalg.inv(sigma2) @ X.T)
    p = first * np.exp(-0.5 * second)
    p = p.reshape(-1, 1) # 转为(n,1)

    return p


def plotGaussian(X, means, sigma2):
    plt.figure()
    x = np.arange(0, 30, 0.5)
    y = np.arange(0, 30, 0.5)
    xx, yy = np.meshgrid(x, y)
    z = gaussian_distribution(np.c_[xx.ravel(), yy.ravel()], means, sigma2)  # 计算对应的高斯分布函数
    zz = z.reshape(xx.shape)
    plt.plot(X[:, 0], X[:, 1], 'bx')
    contour_levels = [10 ** h for h in range(-20, 0, 3)]
    plt.contour(xx, yy, zz, contour_levels)


def select_threshold(yval, p):
    bestEpsilon = 0
    bestF1 = 0
    epsilons = np.linspace(min(p), max(p), 1000)
    for e in epsilons:
        p_ = p < e
        tp = np.sum((yval == 1) & (p_ == 1))
        fp = np.sum((yval == 0) & (p_ == 1))
        fn = np.sum((yval == 1) & (p_ == 0))
        prec = tp / (tp + fp) if (tp + fp) else 0  # precision 精确度
        rec = tp / (tp + fn) if (tp + fn) else 0   # recall召回率
        F1_e = 2 * prec * rec / (prec + rec) if (prec + rec) else 0 # F1score
        if F1_e > bestF1:
            bestF1 = F1_e
            bestEpsilon = e
    return bestEpsilon, bestF1


if __name__ == '__main__':
    mat = sio.loadmat('data/ex8data1.mat')  # dict_keys(['__header__', '__version__', '__globals__', 'X', 'Xval', 'yval'])
    X = mat.get('X')  # X-->(307,2)
    X_val, y_val= mat['Xval'], mat['yval']
    # 原高斯分布模型
    means, sigma2 = estimate_gaussian(X, isCovariance=False)
    plotGaussian(X, means, sigma2)

    # 多元高斯分布模型
    means, sigma2 = estimate_gaussian(X, isCovariance=True)
    plotGaussian(X, means, sigma2)
    # 通过交叉验证集选取阈值ε
    p_val = gaussian_distribution(X_val, means, sigma2)
    bestEpsilon, bestF1 = select_threshold(y_val, p_val)
    print(bestEpsilon, bestF1)

    p = gaussian_distribution(X, means, sigma2)
    anoms = np.array([X[i] for i in range(X.shape[0]) if p[i] < bestEpsilon])
    print(len(anoms))
    print('sss',anoms)
    plotGaussian(X, means, sigma2)
    plt.scatter(anoms[:, 0], anoms[:, 1], c='r', marker='o')

    # plt.plot(X[:, 0], X[:, 1], 'bx')
    plt.show()


# -------------案例2: 高维数据的异常检测  数据集：data/ex8data1.mat-------------------
    mat_h = sio.loadmat('data/ex8data2.mat') # dict_keys(['__header__', '__version__', '__globals__', 'X', 'Xval', 'yval'])
    X2 = mat_h['X']
    X_val2, y_val2 = mat_h['Xval'], mat_h['yval']
    print(X2.shape,X_val2.shape,y_val2.shape)
    means_h,sigma2_h = estimate_gaussian(X2, isCovariance=True)
    pval = gaussian_distribution(X_val2, means_h, sigma2_h)
    bestEpsilon_h, bestF1_h = select_threshold(y_val2, pval)
    p_h = gaussian_distribution(X2,means_h,sigma2_h)
    anoms_h = np.array([X2[i] for i in range(X2.shape[0]) if p_h[i] < bestEpsilon_h])
    print(len(anoms_h))
