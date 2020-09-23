"""
线性--支持向量机
任务：观察C取值对决策边界的影响
数据集：data/ex6data1.mat
"""

import numpy as np
import pandas as pd
import seaborn as sns
sns.set(context="notebook", style='darkgrid', palette='deep')
import scipy.io as sio
import matplotlib.pyplot as plt


def plot_data():
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df['X1'], df['X2'], s=50, c=df['y'], cmap='jet')
    ax.set_title('Raw data')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')


def plot_boundary(model):
    x_min, x_max = -0.5, 4.5
    y_min, y_max = 1.3, 5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))
    z = model.predict(np.c_[xx.flatten(), yy.flatten()])

    zz = z.reshape(xx.shape)
    plot_data()
    plt.contour(xx, yy, zz)


if __name__ == '__main__':
    data = sio.loadmat('./data/ex6data1.mat')
    print(data['X'].shape)
    df = pd.DataFrame({'X1': data['X'][:, 0].flatten(), 'X2': data['X'][:, 1].flatten(), 'y': data['y'].flatten()})
    print(df.head())
    from sklearn.svm import SVC

    # -----------------C=1-------------------------
    clf = SVC(C=1, kernel='linear')
    clf.fit(data['X'], data['y'].flatten())
    y_pred = clf.predict(data['X'])
    print(y_pred, clf.score(data['X'], data['y'].flatten()))
    df['SVM1 Confidence'] = clf.decision_function(df[['X1', 'X2']])
    plot_boundary(clf)

    # -----------------C=100-------------------------
    clf1 = SVC(C=100, kernel='linear')
    clf1.fit(data['X'], data['y'].flatten())
    y_pred = clf.predict(data['X'])
    print(y_pred, clf1.score(data['X'], data['y'].flatten()))
    plot_boundary(clf1)

    df['SVM100 Confidence'] = clf1.decision_function(df[['X1', 'X2']])
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df['X1'], df['X2'], s=50, c=df['SVM100 Confidence'], cmap='autumn')
    ax.set_title('SVM (C=100) Decision Confidence')
    print(df.head())
    plt.show()
