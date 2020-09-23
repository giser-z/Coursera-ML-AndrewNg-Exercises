"""
基于高级优化函数的逻辑回归-正则化
案例：案例：设想你是工厂的生产主管，你要决定是否芯片要被接受或抛弃
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def feature_mapping(x1, x2, power):
    data = {}

    for i in np.arange(power + 1):
        for j in np.arange(i + 1):
            data['F{}{}'.format(i - j, j)] = np.power(x1, i - j) * np.power(x2, j)

    return pd.DataFrame(data)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cost_function(theta, X, y, λ):
    first = y.T @ np.log(sigmoid(X @ theta))
    second = (1 - y.T) @ np.log(1 - sigmoid(X @ theta))
    reg = (λ / (2 * len(X))) * np.sum(np.power(theta[1:], 2))
    # print(first.shape,second.shape,reg)
    return -np.sum(first + second) / len(X) + reg


def gradient(theta, X, y, λ):   # 梯度下降法
    theta = np.mat(theta)
    X = np.mat(X)
    y = np.mat(y)

    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)

    error = sigmoid(X * theta.T) - y
    # print(theta)
    for i in range(parameters):
        term = np.multiply(error, X[:, i])  # X[:, i]--从X中选择第i列数据

        if (i == 0):
            grad[i] = np.sum(term) / len(X)
        else:
            grad[i] = (np.sum(term) / len(X)) + ((λ / len(X)) * theta[:, i])

    return grad


def predict(theta, X):  # theta 为3×1维
    probability = sigmoid(X @ theta)
    return [1 if x >= 0.5 else 0 for x in probability]


if __name__ == "__main__":
    data = pd.read_csv("ex2data2.txt", names=['Test 1', 'Test 2', 'Accepted'])
    fig, ax = plt.subplots()
    ax.scatter(data[data['Accepted'] == 0]['Test 1'], data[data['Accepted'] == 0]['Test 2'], c='r', marker='x',
               label='y=0')
    ax.scatter(data[data['Accepted'] == 1]['Test 1'], data[data['Accepted'] == 1]['Test 2'], c='b', marker='o',
               label='y=1')
    ax.legend()
    ax.set(xlabel='Test1', ylabel='Test2')
    x1 = data['Test 1']
    x2 = data['Test 2']
    data2 = feature_mapping(x1, x2, 6)
    X = np.array(data2.values)
    y = np.array(data.iloc[:, -1].values.reshape(len(X),1))
    #print(X.shape, y.shape)
    theta = np.zeros((28,1))

    import scipy.optimize as opt

    λ = 1
    result = opt.fmin_tnc(func=cost_function, x0=theta, fprime=gradient, args=(X, y, λ))
    print(result[0])
    # 精度验证
    y_ = np.array(predict(result[0], X)).reshape(len(X), 1)
    print(y_.shape, y.shape)  # 注：两个数组维数一定保持完全相同，(118,)与(118,1)不同
    acc = np.mean(y_ == y)
    print('accuracy = {0}'.format(acc))

    # sklearn实现方法
    from sklearn import linear_model  # 调用sklearn的线性回归包

    model = linear_model.LogisticRegression(penalty='l2', C=1.0)
    model.fit(X, y.ravel())
    print("sklerarn_accuracy={}".format(model.score(X, y)))
    # 画图
    x = np.linspace(-1.2, 1.2, 200)
    xx, yy = np.meshgrid(x, x) # 从坐标向量中返回坐标矩阵,例如X轴可以取三个值1,2,3, Y轴可以取三个值7,8, 有坐标(1,7)(2,7)(3,7)(1,8)(2,8)(3,8)
    z = feature_mapping(xx.ravel(), yy.ravel(), 6).values

    zz = z @ result[0]
    zz = zz.reshape(xx.shape)

    fig, ax = plt.subplots()
    ax.scatter(data[data['Accepted'] == 0]['Test 1'], data[data['Accepted'] == 0]['Test 2'], c='r', marker='x',
               label='y=0')
    ax.scatter(data[data['Accepted'] == 1]['Test 1'], data[data['Accepted'] == 1]['Test 2'], c='b', marker='o',
               label='y=1')
    ax.legend()
    ax.set(xlabel='Test1',
           ylabel='Test2')

    plt.contour(xx, yy, zz, 0)
    plt.show()
