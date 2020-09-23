"""
寻找最优参数C和gamma
数据集：data/ex6data3.mat
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import classification_report


def plot_data():
    plt.scatter(X[:,0],X[:,1],c = y.flatten(), cmap ='jet')
    plt.xlabel('x1')
    plt.ylabel('y1')


def plot_boundary(model):
    x_min, x_max = -0.6, 0.4
    y_min, y_max = -0.7, 0.6
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))
    z = model.predict(np.c_[xx.flatten(), yy.flatten()])

    zz = z.reshape(xx.shape)
    plt.contour(xx, yy, zz)

if __name__ =="__main__":
    mat = sio.loadmat('data/ex6data3.mat')
    X, y = mat['X'], mat['y']
    X_val, y_val = mat['Xval'], mat['yval']
    print(X.shape,y.shape)
    plot_data()

    candidate = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
    combination = [(C, gamma) for C in candidate for gamma in candidate]
    print(len(combination))
    search = []
    for C, gamma in combination:
        svc = SVC(C=C, gamma=gamma)
        svc.fit(X, y.flatten())
        search.append(svc.score(X_val, y_val))

    best_score = search[np.argmax(search)] # search最大值--这里有四个相同的最大值(实际上对应四组不同的C，gamma)，默认出现第一次的最大值
    best_param = combination[np.argmax(search)] # combination与search一一对应关系，search最大值对应的索引，也就是combination对应的最佳参数
    print(len(search))

   # best_svc = SVC(C=100, gamma=0.3,kernel="rbf") # 线性分类
    best_svc = SVC(C=0.3, gamma=100, kernel="rbf") # 非线性分类，径向基函数即，核函数为高斯函数
    best_svc.fit(X, y.flatten())
    ypred = best_svc.predict(X_val)

    print(classification_report(y_val.flatten(), ypred))
    plot_boundary(best_svc)
    plt.show()