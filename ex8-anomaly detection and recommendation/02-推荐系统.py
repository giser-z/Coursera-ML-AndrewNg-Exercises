"""
案例：给用户推荐电影
"""

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(context="notebook", style="white", palette=sns.color_palette("RdBu"))
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.optimize as opt


def serialize(X, theta):
    """serialize 2 matrix
    """
    # X (movie, feature), (1682, 10): movie features
    # theta (user, feature), (943, 10): user preference
    return np.append(X.flatten(), theta.flatten())


def deserialize(param, n_movie, n_user, n_features):
    """into ndarray of X(1682, 10), theta(943, 10)"""
    return param[:n_movie * n_features].reshape(n_movie, n_features), \
           param[n_movie * n_features:].reshape(n_user, n_features)


# recommendation fn
def cost_function(param, Y, R, n_features):
    """compute cost for every r(i, j)=1
    Args:
        param: serialized X, theta
        Y (movie, user), (1682, 943): (movie, user) rating
        R (movie, user), (1682, 943): (movie, user) has rating
    """
    # theta (user, feature), (943, 10): user preference
    # X (movie, feature), (1682, 10): movie features
    n_movie, n_user = Y.shape
    X, theta = deserialize(param, n_movie, n_user, n_features)

    inner = np.multiply(X @ theta.T - Y, R)

    return np.power(inner, 2).sum() / 2


def gradient(param, Y, R, n_features):
    # theta (user, feature), (943, 10): user preference
    # X (movie, feature), (1682, 10): movie features
    n_movies, n_user = Y.shape
    X, theta = deserialize(param, n_movies, n_user, n_features)

    inner = np.multiply(X @ theta.T - Y, R)  # (1682, 943)

    # X_grad (1682, 10)
    X_grad = inner @ theta

    # theta_grad (943, 10)
    theta_grad = inner.T @ X

    # roll them together and return
    return serialize(X_grad, theta_grad)


def regularized_cost(param, Y, R, n_features, λ=1):
    reg_term = np.power(param, 2).sum() * (λ / 2)

    return cost_function(param, Y, R, n_features) + reg_term


def regularized_gradient(param, Y, R, n_features, λ=1):
    grad = gradient(param, Y, R, n_features)
    reg_term = λ * param

    return grad + reg_term


def normalizeRatings(Y, R):  # 均值归一化，见算法见笔记16.6
    Y_mean = (Y.sum(axis=1) / R.sum(axis=1)).reshape(-1, 1)
    Y_norm = (Y - Y_mean) * R
    return Y_norm, Y_mean


if __name__ == '__main__':
    movies_mat = sio.loadmat(
        './data/ex8_movies.mat')  # dict_keys(['__header__', '__version__', '__globals__', 'Y', 'R'])
    # Y->(1682,943)(电影数×用户数)用户电影评分矩阵(数组)
    # R->(1682,943) 电影评分的二进制值的"指示符"数据，例如[5 4 0 0 3 0]对应[1 1 0 1 0]
    Y, R = movies_mat.get('Y'), movies_mat.get('R')
    Y_plot = Y

    # param_mat-->dict_keys(['__header__', '__version__', '__globals__', 'X', 'Theta', 'num_users', 'num_movies',
    # 'num_features'])
    param_mat = sio.loadmat('./data/ex8_movieParams.mat')
    theta, X = param_mat.get('Theta'), param_mat.get('X')  # theta->(943,10) 10个特征数量 X->(1682,10) 电影特征(偏好程度)矩阵

    # -------------选取一小段数据，计算代价函数-------------
    users = 4
    movies = 5
    features = 3

    X_sub = X[:movies, :features]
    theta_sub = theta[:users, :features]
    Y_sub = Y[:movies, :users]
    R_sub = R[:movies, :users]
    param_sub = serialize(X_sub, theta_sub)
    cost_sub = cost_function(param_sub, Y_sub, R_sub, features)  # λ=0时的regularized_cost即cost_function函数
    print("小段数据代价函初始值（λ=0）：{}".format(cost_sub))
    cost_sub1 = regularized_cost(param_sub, Y_sub, R_sub, features, λ=1.5)
    print("小段数据代价函初始值（λ=1.5）：{}".format(cost_sub1))

    # -------------整个数据，计算的代价函数-----------------
    param = serialize(X, theta)  # total real params
    print(param.shape)
    cost_real = cost_function(serialize(X, theta), Y, R, 10)  # this is real total cost
    print("所有数据代价函初始值（λ=0）：{}".format(cost_real))
    cost_real_1 = regularized_cost(serialize(X, theta), Y, R, 10, λ=1)
    print("所有数据代价函初始值（λ=1）：{}".format(cost_real_1))

    # 添加一个新的用户，根据该用户偏好自动推荐电影
    nm = int(param_mat['num_movies'])  # 电影数量nm=1682

    new_ratings = np.zeros((nm, 1))
    new_ratings[0] = 4
    new_ratings[6] = 3
    new_ratings[11] = 5
    new_ratings[53] = 4
    new_ratings[63] = 5
    new_ratings[65] = 3
    new_ratings[68] = 5
    new_ratings[97] = 2
    new_ratings[182] = 4
    new_ratings[225] = 5
    new_ratings[354] = 5
    Y = np.c_[Y, new_ratings]  # now I become user 0,在Y最后插入一列，insert貌似只能插入相同数值的列。Y->(1682,944)
    R = np.c_[R, new_ratings != 0]  # R->(1682,944)

    n_movie, n_user = Y.shape  # 电影总数量1682，总用户数量944
    n_features = int(param_mat['num_features'])  # n_features=10
    X_normal = np.random.standard_normal((n_movie, n_features))  # 把数据转换成高斯分布
    theta_normal = np.random.standard_normal((n_user, n_features))
    param_normal = serialize(X_normal, theta_normal)
    # 均值归一化
    Y_norm, Y_mean = normalizeRatings(Y, R)  # Y_mean：电影评分的平均值均，Y_norm：均值归一化结果
    λ = 10

    res = opt.minimize(fun=regularized_cost, x0=param_normal, args=(Y_norm, R, n_features, λ),
                       method='TNC', jac=regularized_gradient)
    print(res)
    X_trained, theta_trained = deserialize(res.x, n_movie, n_user, n_features)
    print(X_trained.shape, theta_trained.shape)
    Y_pred = X_trained @ theta_trained.T
    y_pred = Y_pred[:, -1] + Y_mean.flatten()  # 还原会类似于原始电影评分的数值结果

    index = np.argsort(y_pred)  # 将y_pred中的元素从小到大排序后，提取对应原数组的索引index
    print(index)
    index_reverse = np.flipud(index)  # 数组反转
    movies = []
    with open('data/movie_ids.txt', 'r', encoding='latin 1') as f:
        for line in f:
            tokens = line.strip().split(' ')
            movies.append(' '.join(tokens[1:]))
    print(len(movies))
    # 推荐用户喜好评分较高的电影
    for i in range(10):
        print(index_reverse[i], movies[index_reverse[i]], y_pred[index_reverse[i]])

    # 通过将矩阵渲染成图像来尝试“可视化”数据，我们不能从这里收集太多，但它确实给我们了解用户和电影的相对密度
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(Y, cmap=plt.cm.hot)
    ax.set_xlabel('Users')
    ax.set_ylabel('Movies')
    # ax = sns.heatmap(Y)
    fig.tight_layout()
    plt.show()
