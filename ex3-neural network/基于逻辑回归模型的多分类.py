"""
基于逻辑回归模型的多分类
案例： 手写数字识别
"""


import numpy as np
import scipy.io as sio
import scipy.optimize as opt


def load_data(path, transpose=True):
    data = sio.loadmat(path)
    y = data.get('y')  # (5000,1)
    y = y.reshape(y.shape[0])  # make it back to column vector

    X = data.get('X')  # (5000,400) 每一行是一个数字图(20×20)

    if transpose:
        # for this dataset, you need a transpose to get the orientation right
        X = np.array([im.reshape((20, 20)).T for im in X])  # 把每一行还原为20×20的(正常图片显示)二维数组形式,共5000行，每一行一个二维数组
        # and I flat the image again to preserve the vector presentation
        X = np.array([im.reshape(400) for im in X])

    return X, y


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cost_function(theta, X, y, λ):  # 正则化的代价函数
    first = y.T @ np.log(sigmoid(X @ theta))
    second = (1 - y.T) @ np.log(1 - sigmoid(X @ theta))
    reg = (λ / (2 * len(X))) * np.sum(np.power(theta[1:], 2))
    # print(first.shape,second.shape,reg)
    return -np.sum(first + second) / len(X) + reg


def gradient(theta, X, y, λ):
    theta = np.mat(theta)  # theta变为1×401，矩阵化之前为(401,)，如果是(401,1)就不会发生改变
    X = np.mat(X)  # 如果是矩阵化之前是二维数组形式，格式化后维数不变
    y = np.mat(y)

    error = sigmoid(X * theta.T) - y
    grad = ((X.T * error) / len(X)).T + ((λ / len(X)) * theta)

    # intercept gradient is not regularized
    grad[0, 0] = np.sum(np.multiply(error, X[:, 0])) / len(X)

    return np.array(grad).ravel()


def one_vs_all(X, y, num_labels, λ):
    rows = X.shape[0]
    params = X.shape[1]

    all_theta = np.zeros((num_labels, params))
    # 在X第一列插入1,计算截距(常数项)θ0

    for i in range(1, num_labels + 1):
        theta = np.zeros(params)
        y_i = np.array([1 if label == i else 0 for label in y])
        y_i = np.reshape(y_i, (rows, 1))

        fmin = opt.minimize(fun=cost_function, x0=theta, args=(X, y_i, λ), method='TNC', jac=gradient)
        all_theta[i - 1, :] = fmin.x  # 存入一行(按列方向第0行，1行...排列)
    return all_theta


def predict(X, theta_final):
    # compute the class probability for each class on each training instance
    h = sigmoid(X @ theta_final.T)  # (5000,401) (10,401) =>(5000,10)
    # create array of the index with the maximum probability
    h_argmax = np.argmax(h, axis=1)  # axis=1 按行(每一行最大值的索引)返回最大值索引
    # 返回默认最大索引值是0-9，为了与实际label数据(1-10)对应，对其加1处理
    return h_argmax + 1


if __name__ == "__main__":
    raw_X, raw_y = load_data('ex3data1.mat')  # raw_X--(5000,400),raw_y--(5000,)
    rows = raw_X.shape[0]
    params = raw_X.shape[1]
    # 在X第一列插入1,计算截距(常数项)θ0
    X = np.insert(raw_X, 0, values=np.ones(rows), axis=1)  # axis=1沿行方向添加value=1的列 --插入了第一列（全部为1）,X变为(5000,401)

    theta = np.zeros(params + 1)
    y_0 = np.reshape(raw_y, (rows, 1)) # (5000,1)
    print(y_0.shape)
    print(np.unique(y_0))
    all_theta = one_vs_all(X, y_0, 10, 1)
    print(all_theta.shape)
    # 精度验证
    y_pred = predict(X, all_theta)
    y_pred = y_pred.reshape((rows, 1))  # 注：两个数组(预测的label与真实label)维数一定保持完全相同
    print(y_pred.shape)
    print('Accuracy={}'.format(np.mean(y_0 == y_pred)))
