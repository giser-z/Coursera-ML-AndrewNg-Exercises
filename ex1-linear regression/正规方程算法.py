"""
线性回归-正规方程法

"""

import numpy as np
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
    return np.array(df.iloc[:, -1])  # 返回df最后一列


def normal_equation(X, y):
    theta = np.linalg.inv(X.T @ X) @ X.T @ y  # np.linalg.inv(X.T@X)矩阵的逆
    return theta


if __name__ == "__main__":
    data = pd.read_csv('ex1data1.txt', names=['population', 'profit'])
    X = get_X(data)
    y = get_y(data)
    theta = normal_equation(X, y)
    print(theta)
