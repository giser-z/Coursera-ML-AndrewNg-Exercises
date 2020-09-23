"""
基于神经网络的多分类问题
案例： 手写数字识别
"""
import numpy as np
import scipy.io as sio
from sklearn.metrics import classification_report # 这个包是评价报告


def load_data(path, transpose=True):
    data = sio.loadmat(path)
    y = data.get('y')  # (5000,1)
    y = y.reshape(y.shape[0])  # make it back to column vector

    X = data.get('X')  # (5000,400) 每一行是一个数字图(20×20)

    if transpose:
        # for this dataset, you need a transpose to get the orientation right
        # X = np.array([im.reshape((20, 20)).T for im in X])  # 把每一行还原为20×20的(正常图片显示)二维数组形式,共5000行，每一行一个二维数组
        # and I flat the image again to preserve the vector presentation
        X = np.array([im.reshape(400) for im in X])

    return X, y


def load_weight(path):
    data = sio.loadmat(path)
    return data["Theta1"], data["Theta2"]


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


if __name__ == "__main__":
    raw_X, raw_y = load_data('ex3data1.mat')  # raw_X--(5000,400),raw_y--(5000,)

    X = np.insert(raw_X, 0, values=np.ones(raw_X.shape[0]), axis=1)  # 插入了第一列（全部为1）,X变为(5000,401)
    theta1, theta2 = load_weight("ex3weights.mat")  # theta1为(25,401),theta2为(10,26)
    a1 = X
    z2 = a1 @ theta1.T  # (5000, 401) @ (25,401).T = (5000, 25)
    a2 = sigmoid(z2)
    a2 = np.insert(a2, 0, values=1, axis=1) # (5000,26)
    print(a2.shape)
    z3 = a2 @ theta2.T  # (5000,26) @ (26,10) = (5000,10)
    a3 = sigmoid(z3)
    y_pred = np.argmax(a3, axis=1)
    y_pred = y_pred + 1
    print(y_pred)
    print('Accuracy={}'.format(np.mean(raw_y == y_pred)))

    print(classification_report(raw_y, y_pred))
