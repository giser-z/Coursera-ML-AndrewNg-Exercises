"""
非线性--支持向量机
任务：使用高斯核函数解决线性不可分问题，并观察gamma取值对模型复杂度的影响
数据集：data/ex6data2.mat

"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.svm import SVC


def plot_data():
    plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), cmap='jet')
    plt.xlabel('x1')
    plt.ylabel('y1')


def plot_boundary(model):
    x_min, x_max = 0, 1
    y_min, y_max = 0.4, 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500)) # 把x,y数据生成mesh网格状的数据，因为等高线的显示是在网格的基础上添加上高度值
    z = model.predict(np.c_[xx.flatten(), yy.flatten()])
    print(z.shape)
    zz = z.reshape(xx.shape)
    plot_data()
    plt.contour(xx, yy, zz)

if __name__ == "__main__":
    data = sio.loadmat('./data/ex6data2.mat')
    X, y = data['X'], data['y']
    # print(data)
    plot_data()

    clf = SVC(C=1, kernel='rbf', gamma=100)
    clf.fit(X, y.flatten())
    print(clf.score(X, y.flatten()))
    plot_boundary(clf)
    plt.show()
