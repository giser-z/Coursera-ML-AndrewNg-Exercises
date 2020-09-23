"""
逻辑回归
案例：根据学生的两门学生成绩，预测该学生是否会被大学录取
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_Xy(data):
    data.insert(0, 'ones', 1)
    X = np.array(data.iloc[:, 0:-1])
    y = np.array(data.iloc[:, -1])
    return X, y


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cost_function(theta, X, y): # theta 为3×1维
    first = y * np.log(sigmoid(X @ theta))  # first为(100,)维度为1，长度100,*点乘，对数组执行对应位置相乘，对矩阵执行矩阵乘法运算
    second = (1 - y) * np.log(1 - sigmoid(X @ theta))
    #print(first.shape,second.shape)
    return -np.sum(first + second) / len(X)


def gradient_descent(X, y, theta, epoch, alpha): # theta 为3×1维
    m = len(X)
    costs = []

    for i in range(epoch):
        A = sigmoid(X @ theta)
        theta = theta - (alpha / m) * X.T @ (A - y)
        cost = cost_function(theta, X, y)
        costs.append(cost)
        if i % 1000 == 0:
            print(cost)
    return costs, theta


def gradient(theta, X, y): # 迭代了一次的梯度  theta 为3×1维
    parameters = int(theta.ravel().shape[0])  # ravel展平数组
    grad = np.zeros(parameters)  # grad赋与theta一样的维度，3×1
    grad = grad.T
    error = sigmoid(X @ theta) - y

    for i in range(parameters):
        term = np.multiply(error, X[:, i])
        grad[i] = np.sum(term) / len(X)

    return grad


def predict(theta, X): # theta 为3×1维
    probability = sigmoid(X @ theta)
    return [1 if x >= 0.5 else 0 for x in probability]


if __name__ == "__main__":
    data = pd.read_csv("ex2data1.txt", names=['Exam 1', 'Exam 2', 'Admitted'])
    positive = data[data['Admitted'].isin([1])]
    negative = data[data['Admitted'].isin([0])]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(positive['Exam 1'], positive['Exam 2'], s=50, c='b', marker='o',
               label='Admitted')  # s 浮点或数组形式，shape(n,),可选大小以点数平方。c表示颜色
    ax.scatter(negative['Exam 1'], negative['Exam 2'], s=50, c='r', marker='x', label='Not Admitted')
    # 也可以用如下方法
    # ax.scatter(data[data['Accepted']==0]['Exam 1'],data[data['Accepted']==0]['Exam 2'],c='r',marker='x',label='y=0')
    # ax.scatter(data[data['Accepted']==1]['Exam 1'],data[data['Accepted']==1]['Exam 2'],c='b',marker='o',label='y=1')
    ax.legend()
    ax.set_xlabel('Exam 1 Score')
    ax.set_ylabel('Exam 2 Score')
    X, y = get_Xy(data)
    print(X.shape,y.shape)
    theta = np.zeros(3,)  # 1×3维
    θ = theta.T
    # *****************选择学习速率α******************************
    cost_init = cost_function(θ, X, y)
    print("初始最小代价函数值：{}".format(cost_init))
    epoch = 200000
    alpha = 0.004
    costs, final_theta = gradient_descent(X, y, θ, epoch, alpha)
    print(final_theta)
    # 精度验证
    y_ = np.array(predict(final_theta, X))
    print(y_.shape,y.shape)
    acc = np.mean(y_ == y)
    print ('accuracy = {0}'.format(acc))
    print("-" * 30, "我是分割线", "-" * 30)
    # *****************调用高级优化函数--自动选择学习速率α******************************
    import scipy.optimize as opt

    result = opt.fmin_tnc(func=cost_function, x0=θ, fprime=gradient, args=(X, y))
    print(result)
    print("最终代价函数计算结果：{}".format(cost_function(result[0], X, y)))
    # 精度验证
    y_1 = np.array(predict(result[0], X))
    print(y_1.shape, y.shape)
    acc1 = np.mean(y_1 == y)
    print ('accuracy_1 = {0}'.format(acc1))

    plt.show()
