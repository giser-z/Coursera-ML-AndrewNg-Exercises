"""
多变量线性回归
案例：假设你现在打算卖房子，想知道房子能卖多少钱？
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def normalize_feature(df):
    '''标准化数据(特征缩放)'''
    return (df - df.mean()) / df.std()


# 散点图
# data_nomalized.plot.scatter('size','price',label='size')
# data_nomalized.plot.scatter('bedrooms','price',label='bedrooms')
# plt.show()
def get_X(df):
    """
       读取特征
       use concat to add intersect feature to avoid side effect
       not efficient for big dataset though
    """
    ones = pd.DataFrame({'ones': np.ones(len(df))})  # len(data)行的长度,插入标签ones一列，赋值为1 创建m×1的矩阵(dataframe)
    data = pd.concat([ones, df], axis=1)  # 将创建的ones合并入df中
    # 以上两句代码或者直接用data.insert('位置索引(如0)',"标签名(ones)",'赋值(1)')插入
    # print(data)
    return data.iloc[:, 0:2]  # loc是自定义索引标签，iloc是默认索引标签(0,1,2...) iloc[行，列]


def get_y(df):
    """ 读取标签
    assume the last column is the target
    """
    return np.array(df.iloc[:, -1])  # 返回df最后一列


def cost_function(theta, X, y):
    m = X.shape[0]  # 样本数量
    inner = X @ theta - y
    square_num = inner.T @ inner
    cost = square_num / (2 * m)
    return cost


def batch_gradient_descent(theta, X, y, epoch, alpha=0.01):
    """
    拟合线性回归，返回参数和代价
         epoch: 批处理的轮数
    """
    m = X.shape[0]
    cost_data = [cost_function(theta, X, y)]  # 将theta为0时，最小代价函数加入到列表
    for _ in range(epoch):
        theta = theta - (X.T @ (X @ theta - y)) * alpha / m
        cost_data.append(cost_function(theta, X, y))
    return theta, cost_data  # 返回最后的theta和保存每次迭代最小化代价函数值列表


if __name__ == "__main__":
    data = pd.read_csv('ex1data2.txt', names=['size', 'bedrooms', 'price'])
    data_nomalized = normalize_feature(data)
    # print(data_nomalized.head())
    X = get_X(data_nomalized)
    y = get_y(data_nomalized)
    alpha = 0.01  # 学习率
    theta = np.zeros(X.shape[1])  # X.shape[1]：特征数n
    epoch = 500  # 轮数
    final_theta, cost_data = batch_gradient_descent(theta, X, y, epoch)

    # 最小代价数据可视化
    fig = plt.figure(figsize=(12, 4))  # 表示figure 的大小为宽、长（单位为inch）
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(len(cost_data)), cost_data)
    plt.xlabel('epoch')
    plt.ylabel('loss')

    # 不同学习速率alpha的效果
    base = np.logspace(-1, -5, 4)  # np.logspace(start=开始值，stop=结束值，num=元素个数，base=指定对数的底(默认底数10), endpoint=是否包含结束值)
    candidate = np.sort(
        np.concatenate((base, base * 3), axis=0))  # sort axis=0按列排序(对每一列排序)，默认按行(axis=1)排序。concatenate默认axis=0(沿列方向),沿轴连接
    print(candidate)
    epoch = 50
    ax = plt.subplot(1, 2, 2)
    for alpha in candidate:
        _, cost_data = batch_gradient_descent(theta, X, y, epoch, alpha=alpha)
        ax.plot(np.arange(len(cost_data)), cost_data, label=alpha)

    ax.set_xlabel('epoch', fontsize=10)
    ax.set_ylabel('cost', fontsize=10)
    ax.legend(bbox_to_anchor=(0.5, 1), loc=2, borderaxespad=0)  # bbox_to_anchor调节图例位置
    ax.set_title('learning rate', fontsize=10)
    plt.show()
