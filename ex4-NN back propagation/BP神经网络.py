"""
神经网络-反向传播多分类问题
案例： 手写数字识别
"""

import numpy as np
import scipy.io as sio
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.metrics import classification_report # 这个包是评价报告


def load_data(path, transpose=True):
    data = sio.loadmat(path)
    y = data.get('y')  # (5000,1)
    y = y.reshape(y.shape[0])  # make it back to column vector

    X = data.get('X')  # (5000,400) 每一行是一个数字图(20×20)

    if transpose:
        # for this dataset, you need a transpose to get the orientation right
        # X = np.array([im.reshape((20, 20)).T for im in X])  # 把每一行还原为20×20的二维数组形式,共5000行，每一行一个二维数组
        # and I flat the image again to preserve the vector presentation
        X = np.array([im.reshape(400) for im in X])

    return X, y


def load_weight(path):
    data = sio.loadmat(path)
    return data["Theta1"], data["Theta2"]


def expand_y(y):
    """expand 5000*1 into 5000*10
         where y=10 -> [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]: ndarray

    from sklearn.preprocessing import OneHotEncoder
    encoder = OneHotEncoder(sparse=False)
    y_onehot = encoder.fit_transform(y)
    y_onehot.shape #这个函数与expand_y(y)一致
    """
    res = []
    for i in y:  # y分类标签1-10
        y_array = np.zeros(10)
        y_array[i - 1] = 1

        res.append(y_array)

    return np.array(res)


def serialize(a, b):
    return np.append(a.flatten(), b.flatten())


def deserialize(theta_serialize):
    """into ndarray of (25, 401), (10, 26)"""
    theta1 = theta_serialize[:25 * 401].reshape(25, 401)
    theta2 = theta_serialize[25 * 401:].reshape(10, 26)
    return theta1, theta2


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def feed_forward(theta_serialize, X):  # 前向传播
    theta1, theta2 = deserialize(theta_serialize)
    a1 = X
    z2 = a1 @ theta1.T
    a2 = sigmoid(z2)
    a2 = np.insert(a2, 0, values=1, axis=1)
    z3 = a2 @ theta2.T
    h = sigmoid(z3)
    return a1, z2, a2, z3, h


def cost(theta_serialize, X, y):  # 不带正则化的损失函数
    a1, z2, a2, z3, h = feed_forward(theta_serialize, X)
    J = -np.sum(y * np.log(h) + (1 - y) * np.log(1 - h)) / len(X)
    return J


def regularized_cost(theta_serialize, X, y, λ=1):  # 带正则化的损失函数
    """the first column of t1 and t2 is intercept theta, ignore them when you do regularization"""
    t1, t2 = deserialize(theta_serialize)  # t1: (25,401) t2: (10,26)
    m = X.shape[0]

    reg_t1 = (λ / (2 * m)) * np.power(t1[:, 1:], 2).sum()  # this is how you ignore first col
    reg_t2 = (λ / (2 * m)) * np.power(t2[:, 1:], 2).sum()

    return cost(theta_serialize, X, y) + reg_t1 + reg_t2


def sigmoid_gradient(z): # sigmoid函数的导数实现
    return sigmoid(z) * (1 - sigmoid(z))


def gradient(theta_serialize, X, y):  # 无正则化梯度
    theta1, theta2 = deserialize(theta_serialize)
    a1, z2, a2, z3, h = feed_forward(theta_serialize, X)
    d3 = h - y
    d2 = d3 @ theta2[:, 1:] * sigmoid_gradient(z2)
    D2 = (d3.T @ a2) / len(X)
    D1 = (d2.T @ a1) / len(X)
    return serialize(D1, D2)


def reg_gradient(theta_serialize, X, y, λ=1):  # 带正则化梯度
    D = gradient(theta_serialize, X, y)
    D1, D2 = deserialize(D)

    theta1, theta2 = deserialize(theta_serialize)
    D1[:, 1:] = D1[:, 1:] + theta1[:, 1:] * λ / len(X)
    D2[:, 1:] = D2[:, 1:] + theta2[:, 1:] * λ / len(X)

    return serialize(D1, D2)


def nn_training(X, y, λ=1): # 训练模型
    init_theta = np.random.uniform(-0.12, 0.12, 10285) # 随机初始化theta值
    res = minimize(fun=regularized_cost,
                   x0=init_theta,
                   args=(X, y, λ),
                   method='TNC',
                   jac=reg_gradient,
                   options={'maxiter': 400})

    return res


def plot_hidden_layer(theta):  # 可视化隐藏层
    theta1, _ = deserialize(theta)
    hidden_layer = theta1[:, 1:]  # 25,400  去掉偏差单元

    fig, ax = plt.subplots(ncols=5, nrows=5, figsize=(8, 8), sharex=True, sharey=True)

    for r in range(5):
        for c in range(5):
            ax[r, c].matshow(hidden_layer[5 * r + c].reshape((20, 20)), cmap=matplotlib.cm.binary)

    plt.xticks([])
    plt.yticks([])


# -------梯度校验---------
def expand_array(arr):
    """replicate array into matrix
    [1, 2, 3]

    [[1, 2, 3],
     [1, 2, 3],
     [1, 2, 3]]
    """
    # turn matrix back to ndarray
    return np.array(np.mat(np.ones(arr.shape[0])).T @ np.mat(arr))


def gradient_checking(theta, X, y, epsilon, regularized=False):
    def a_numeric_grad(plus, minus, regularized=False):
        """calculate a partial gradient with respect to 1 theta"""
        if regularized:
            return (regularized_cost(plus, X, y) - regularized_cost(minus, X, y)) / (epsilon * 2)
        else:
            return (cost(plus, X, y) - cost(minus, X, y)) / (epsilon * 2)

    theta_matrix = expand_array(theta)  # expand to (10285, 10285)
    epsilon_matrix = np.identity(len(theta)) * epsilon

    plus_matrix = theta_matrix + epsilon_matrix
    minus_matrix = theta_matrix - epsilon_matrix

    # calculate numerical gradient with respect to all theta
    numeric_grad = np.array([a_numeric_grad(plus_matrix[i], minus_matrix[i], regularized)
                             for i in range(len(theta))])

    # analytical grad will depend on if you want it to be regularized or not
    analytic_grad = reg_gradient(theta, X, y) if regularized else gradient(theta, X, y)

    # If you have a correct implementation, and assuming you used EPSILON = 0.0001
    # the diff below should be less than 1e-9
    # this is how original matlab code do gradient checking
    diff = np.linalg.norm(numeric_grad - analytic_grad) / np.linalg.norm(numeric_grad + analytic_grad)

    print(
        'If your backpropagation implementation is correct,\nthe relative difference will be smaller than 10e-9 (assume epsilon=0.0001).\nRelative Difference: {}\n'.format(
            diff))


if __name__ == '__main__':
    X_raw, y_raw = load_data('ex4data1.mat', transpose=False)
    X = np.insert(X_raw, 0, np.ones(X_raw.shape[0]), axis=1)  # 增加全部为1的一列
    theta1, theta2 = load_weight('ex4weights.mat') # theta1->(25,401), theta2->(10,26)
    y = expand_y(y_raw) # y-->(5000,10),标签分类范围0-9
    theta = serialize(theta1, theta2)  # 扁平化参数，25*401+10*26=10285,(10285,)
    print(theta.shape)
    print(regularized_cost(theta, X, y))

    # 反向传播
    res = nn_training(X, y, 1)
    _, _, _, _, h = feed_forward(res.x, X)

    y_pred = np.argmax(h, axis=1) + 1
    print('Accuracy={}'.format(np.mean(y_raw == y_pred)))
    plot_hidden_layer(res.x)  # 显示的是第一组θ的值
    # gradient_checking(res.x, X, y, epsilon=0.0001)  # 这个运行很慢，谨慎运行,占用很大内存
    # If your backpropagation implementation is correct,
    # the relative difference will be smaller than 10e-9 (assume epsilon=0.0001).
    # Relative Difference: 2.1455623285988868e-09

    # 混淆矩阵精度验证
    print(classification_report(y_raw, y_pred))

    plt.show()
