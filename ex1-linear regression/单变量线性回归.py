"""
单变量线性回归
案例：假设你是一家餐厅的CEO，正在考虑开一家分店，根据该城市的人口数据预测其利润。
"""
import numpy  as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


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
    return np.array(df.iloc[:, -1])  # 返回df最后一列--dataframe转np数组


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
    data = pd.read_csv('ex1data1.txt', names=['population', 'profit'])

    sns.set(context='notebook', style='whitegrid', palette='dark')
    # print('info:',data.info(),'\n','head:','\n',data.head(),'\n','tail:','\n',data.tail(),'\n','describe:','\n',data.describe())
    # sns.lmplot('population','profit',data,height=6,fit_reg=False)
    # plt.show()
    # data.plot.scatter('population','profit',label='population')
    # # plt.show()

    X = get_X(data)  # (97, 2) <class 'numpy.ndarray'>
    # print(X.shape,type(X))
    # print("*"*20,"\n",X)
    y = get_y(data)   # 97×1向量(列表)
    theta = np.zeros(X.shape[1])  # X.shape=(97,2),代表特征数量，theta初始值赋0
    cost_num = cost_function(theta, X, y)
    # print(cost_num)
    epoch = 2000
    alpha = 0.02
    final_theta, cost_data = batch_gradient_descent(theta, X, y, epoch)
    print("theta值：\n",final_theta)
    print("每次迭代的最小代价函数值：",cost_data)
    final_cost = cost_function(final_theta, X, y) # 计算最终代价函数值
    print("最终代价函数值：",final_cost)

    # 最小代价数据可视化
    plt.figure(figsize=(12, 4)) # 表示figure 的大小为宽、长（单位为inch）
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(epoch+1),cost_data)
    plt.xlabel('epoch')
    plt.ylabel('loss')

    plt.subplot(1, 2, 2)
    plt.scatter(data.population, data.profit, label="Training data")
    plt.plot(data.population, data.population*final_theta[1] + final_theta[0], label="prediction",color="#FF0000")
    plt.legend(loc=2)
    plt.show()


