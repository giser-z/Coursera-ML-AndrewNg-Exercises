"""
逻辑回归-正则化
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
    reg = (λ / (2 * len(X))) * np.sum(np.power(theta[1:], 2)) # 排除theta0--theta[1:]
    # print(first.shape,second.shape,reg)
    return -np.sum(first + second) / len(X) + reg


def gradient_descent(theta, X, y, α, epoch, λ):
    costs = []

    for i in range(epoch):

        reg = theta[1:] * (λ / len(X))
        reg = np.insert(reg, 0, values=0, axis=0)

        theta = theta - (X.T @ (sigmoid(X @ theta) - y)) * α / len(X) - reg
        cost = cost_function(theta, X, y, λ)
        costs.append(cost)

        if i % 1000 == 0:
            print(cost)

    return theta, costs


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
    y = np.array(data.iloc[:, -1].values).reshape(len(X), 1)
    print(X.shape, y.shape)

    theta = np.zeros((28, 1))
    cost_init = cost_function(theta, X, y, λ=1)
    print("初始最小代价函数值：{}".format(cost_init))
    α = 0.001
    epoch = 200000
    final_theta, costs = gradient_descent(theta, X, y, α, epoch, λ=0.1)
    print("final_theta：{}".format(final_theta))
    # 精度验证
    y_ = np.array(predict(final_theta, X)).reshape(len(X), 1)
    print(y_.shape, y.shape) # 注：两个数组维数一定保持完全相同，(118,)与(118,1)不同
    acc = np.mean(y_ == y)
    print('accuracy = {0}'.format(acc))

    plt.show()
