"""
了解算法性能中的偏差和方差概念
案例：利用水库水位变化预测大坝出水量
数据集：ex5data1.mat
机器学习课程的第五个编程练习（第六周讲解内容）
"""

import numpy as np
import scipy.io as sio
import scipy.optimize as opt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def reg_cost(theta, X, y, λ):
    cost = np.sum(np.power((X @ theta - y.flatten()), 2))
    reg = theta[1:] @ theta[1:] * λ

    return (cost + reg) / (2 * len(X))


def reg_gradient(theta, X, y, λ):
    grad = (X @ theta - y.flatten()) @ X
    reg = λ * theta
    reg[0] = 0

    return (grad + reg) / (len(X))


def linear_regression_model(X, y, λ):
    theta = np.ones(X.shape[1])

    res = opt.minimize(fun=reg_cost,
                       x0=theta,
                       args=(X, y, λ),
                       method='TNC',
                       jac=reg_gradient)

    return res.x


def plot_learning_curve(X_train, y_train, X_val, y_val, λ): # 构造样品数量与误差曲线
    x = range(1, len(X_train) + 1)
    training_cost = []
    cv_cost = []
    for i in x:
        res = linear_regression_model(X_train[:i, ], y_train[:i, :], λ)
        training_cost_i = reg_cost(res, X_train[:i, ], y_train[:i, ], λ)
        cv_cost_i = reg_cost(res, X_val, y_val, λ)
        training_cost.append(training_cost_i)
        cv_cost.append(cv_cost_i)
    fig, ax = plt.subplots()
    ax.plot(x, training_cost, label='traning cost')
    ax.plot(x, cv_cost, label='cross_validation cost')
    ax.legend()
    plt.xlabel('number of training examples')
    plt.ylabel('error')


# *****************构造多项式特征****************************
def poly_feature(X, power):
    for i in range(2, power + 1):
        X = np.insert(X, X.shape[1], np.power(X[:, 1], i), axis=1)
    return X


def get_means_stds(X):
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)
    return means, stds


def feature_normalize(X, means, stds):
    X[:, 1:] = (X[:, 1:] - means[1:]) / stds[1:]

    return X


def plot_data():  # 绘制散点图
    fig, ax = plt.subplots()
    ax.scatter(X_train[:, 1], y_train)
    ax.set(xlabel='change in water level(x)',
           ylabel='water flowing out og the dam(y)')


def plot_poly_fit():
    plot_data()

    x_axis = np.linspace(-60, 60, 100)
    x = x_axis.reshape(100, 1)
    x = np.insert(x, 0, 1, axis=1)
    x = poly_feature(x, power)
    x = feature_normalize(x, train_means, train_stds)
    plt.plot(x_axis, x @ theta_fit, 'r--')


if __name__ == "__main__":
    data = sio.loadmat('ex5data1.mat')
    X_train, y_train = data['X'], data['y']  # (12,1)(12,1)
    X_test, y_test = data['Xtest'], data['ytest']
    print(X_train, y_train.shape)
    df = pd.DataFrame({'water_level': np.reshape(X_train, len(X_train)), 'flow': np.reshape(y_train, len(y_train))})

    # -----------------散点图与线性拟合线(Figure1)-------------------------
    sns.set(context="notebook", style='darkgrid', palette='deep')
    sns.lmplot('water_level', 'flow', data=df, fit_reg=False, height=5)

    X_train = np.insert(X_train, 0, 1, axis=1)  # X_train-->(12,2)
    theta = np.ones(X_train.shape[1])
    # X_train = np.insert(X, 0, 1, axis=1)

    print(reg_cost(theta, X_train, y_train, λ=0))
    print(reg_gradient(theta, X_train, y_train, λ=0))
    final_teta = linear_regression_model(X_train, y_train, λ=0)
    plt.plot(X_train[:, 1], X_train @ final_teta, c='r')  # 拟合线

    # --------------------样本数量与误差的变化曲线(Figure2)----------------
    X_val, y_val = data['Xval'], data['yval']
    X_val = np.insert(X_val, 0, 1, axis=1)
    plot_learning_curve(X_train, y_train, X_val, y_val, λ=0)

    # -------------构造多项式特征，进行多项式回归拟合(Figure3-Figure6)-------------------
    power = 6
    X_test = np.insert(X_test, 0, 1, axis=1)
    X_train_poly = poly_feature(X_train, power)
    X_val_poly = poly_feature(X_val, power)
    X_test_poly = poly_feature(X_test, power)

    train_means, train_stds = get_means_stds(X_train_poly)  # 每一列(特征)的均值和均方根误差

    X_train_norm = feature_normalize(X_train_poly, train_means, train_stds)
    X_val_norm = feature_normalize(X_val_poly, train_means, train_stds)
    X_test_norm = feature_normalize(X_test_poly, train_means, train_stds)

    theta_fit = linear_regression_model(X_train_norm, y_train, λ=0)
    plot_poly_fit() # Figure3
    plot_learning_curve(X_train_norm, y_train, X_val_norm, y_val, λ=0) # Figure4 过拟合

    theta_fit = linear_regression_model(X_train_norm, y_train, λ=100)
    plot_poly_fit() # Figure5
    plot_learning_curve(X_train_norm, y_train, X_val_norm, y_val, λ=100) # Figure6正则化系数过大，变成欠拟合了

    # -------------------------找最佳的λ Figure7------------------------------
    λ_values = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
    training_cost, cv_cost = [], []
    for λ in λ_values:
        res_x = linear_regression_model(X_train_norm, y_train, λ)

        tc = reg_cost(res_x, X_train_norm, y_train, 0)
        cv = reg_cost(res_x, X_val_norm, y_val, 0)

        training_cost.append(tc)
        cv_cost.append(cv)

    fig_1, ax1 = plt.subplots()
    plt.plot(λ_values, training_cost, label='training cost')
    plt.plot(λ_values, cv_cost, label='cv cost')
    plt.legend(loc=2)

    plt.xlabel('λ')
    plt.ylabel('cost')
    λ_fit = λ_values[np.argmin(cv_cost)]
    print(λ_fit) # 调参后，  𝜆=0.3  是最优选择，这个时候测试代价最小
    # use test data to compute the cost--测试集-
    for λ in λ_values:
        theta_ = linear_regression_model(X_train_norm, y_train, λ)
        print('test cost(λ={}) = {}'.format(λ, reg_cost(theta_, X_test_norm, y_test, 0)))

    plt.show()
