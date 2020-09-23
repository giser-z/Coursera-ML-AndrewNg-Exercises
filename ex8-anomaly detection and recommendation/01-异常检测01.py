"""
案例1: 检测异常服务器
数据集：data/ex8data1.mat
注：多元高斯分布模型算法借助其他库(scipy)来实现
"""

import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(context="notebook", style="white", palette=sns.color_palette("RdBu"))
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report


def select_threshold(X, X_val, y_val):
    """use CV data to find the best epsilon
    Returns:
        e: best epsilon with the highest f-score
        f-score: such best f-score
    """
    # create multivariate model using training data
    mu = X.mean(axis=0)
    cov = np.cov(X.T)
    multi_normal = stats.multivariate_normal(mu, cov)

    # this is key, use CV data for fine tuning hyper parameters
    pval = multi_normal.pdf(X_val)

    # set up epsilon candidates
    epsilon = np.linspace(np.min(pval), np.max(pval), num=10000)

    # calculate f-score
    fs = []
    for e in epsilon:
        y_pred = (pval <= e).astype('int')
        fs.append(f1_score(y_val, y_pred))

    # find the best f-score
    argmax_fs = np.argmax(fs)

    return epsilon[argmax_fs], fs[argmax_fs]


def predict(X, X_val, e, X_test, y_test):
    """with optimal epsilon, combine X, Xval and predict Xtest
    Returns:
        multi_normal: multivariate normal model
        y_pred: prediction of test data
    """
    Xdata = np.concatenate((X, X_val), axis=0)

    mu = Xdata.mean(axis=0)
    cov = np.cov(Xdata.T)
    multi_normal = stats.multivariate_normal(mu, cov)

    # calculate probability of test data
    pval = multi_normal.pdf(X_test)
    y_pred = (pval <= e).astype('int')

    print(classification_report(y_test, y_pred))

    return multi_normal, y_pred


if __name__ == '__main__':
    mat = sio.loadmat(
        'data/ex8data1.mat')  # dict_keys(['__header__', '__version__', '__globals__', 'X', 'Xval', 'yval'])
    X = mat.get('X')  # X-->(307,2)
    # 该方法与前一种方法，在数据有区别，前一种没有划分测试集数据，该方法划分出测试集数据，
    X_val, X_test, y_val, y_test = train_test_split(mat.get('Xval'), mat.get('yval').ravel(), test_size=0.5)

    e, fs = select_threshold(X, X_val, y_val)
    print('Best epsilon: {}\nBest F-score on validation data: {}'.format(e, fs))
    multi_normal, y_pred = predict(X, X_val, e, X_test, y_test)

    # create a grid
    x, y = np.mgrid[0:30:0.5, 0:30:0.5]  # x->(3000,3000),y->(3000,3000)
    pos = np.dstack((x, y))
    print(pos.shape)

    fig, ax = plt.subplots()
    # plot probability density
    ax.contourf(x, y, multi_normal.pdf(pos), cmap='Blues')

    # plot original data points
    sns.regplot('Latency', 'Throughput',
                data=pd.DataFrame(X, columns=['Latency', 'Throughput']),
                fit_reg=False,
                ax=ax,
                scatter_kws={"s": 10,
                             "alpha": 0.4})

    # construct test DataFrame
    data = pd.DataFrame(X_test, columns=['Latency', 'Throughput'])
    data['y_pred'] = y_pred
    # mark the predicted anamoly of CV data. We should have a test set for this...
    anamoly_data = data[data['y_pred'] == 1]
    ax.scatter(anamoly_data['Latency'], anamoly_data['Throughput'], marker='x', s=50, c='r')
    print(anamoly_data.shape)
    plt.show()

# -----------------------------------high dimension data-------------------------------
    mat_h = sio.loadmat('./data/ex8data2.mat')
    X_h = mat_h.get('X')
    Xval, Xtest, yval, ytest = train_test_split(mat_h.get('Xval'), mat_h.get('yval').ravel(), test_size=0.5)
    e_h, fs_h = select_threshold(X_h, Xval, yval)
    print('Best epsilon: {}\nBest F-score on validation data: {}'.format(e_h, fs_h))
    multi_normal_h, y_pred_h = predict(X_h, Xval, e_h, Xtest, ytest)
    print('find {} anamolies'.format(y_pred_h.sum()))
