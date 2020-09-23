"""
案例： 手写数字识别
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import matplotlib


def load_data(path, transpose=True):
    data = sio.loadmat(path)
    y = data.get('y')  # (5000,1)
    y = y.reshape(y.shape[0])  # make it back to column vector

    X = data.get('X')  # (5000,400) 每一行是一个数字图(20×20)
    print(X.shape, y.shape)
    if transpose:
        # for this dataset, you need a transpose to get the orientation right
        X = np.array([im.reshape((20, 20)).T for im in X])  # 把每一行还原为20×20的(正常图片显示)二维数组形式,共5000行，每一行一个二维数组
        # and I flat the image again to preserve the vector presentation
        X = np.array([im.reshape(400) for im in X])
        print(len([1 if label == 0 else 0 for label in y]))

    return X, y


def plot_an_image(image):  # 绘图函数
    #     """
    #     image : (400,)
    #     """
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.matshow(image.reshape((20, 20)),
               cmap=matplotlib.cm.binary)  # matshow()函数绘制矩阵,cmap意思是color map，颜色方案，binary代表是白底黑字
    plt.xticks(np.array([]))  # just get rid of ticks
    plt.yticks(np.array([]))  # 一个是刻标(locs)，一个是刻度标签(tick labels),不显示tick可以传入空的参数(不显示刻度)


def plot_100_image(X):  # 绘图函数，画100张图片
    """ sample 100 image and show them
    assume the image is square

    X : (5000, 400)
    """
    size = int(np.sqrt(X.shape[1]))

    # sample 100 image, reshape, reorg it
    sample_idx = np.random.choice(np.arange(X.shape[0]), 100)  # 100*400 5000张图片中取出100张(5000行里随机取100行)
    sample_images = X[sample_idx, :]

    fig, ax_array = plt.subplots(nrows=10, ncols=10, sharey=True, sharex=True, figsize=(8, 8))  # sharey=True共享x,y轴

    for r in range(10):
        for c in range(10):
            ax_array[r, c].matshow(sample_images[10 * r + c].reshape((size, size)),
                                   cmap=matplotlib.cm.binary)
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))


if __name__ == '__main__':
    X, y = load_data('ex3data1.mat')
    pick_one = np.random.randint(0, 5000)
    plot_100_image(X)
    plot_an_image(X[pick_one, :])
    plt.show()
    print('this should be {}'.format(y[pick_one]))
