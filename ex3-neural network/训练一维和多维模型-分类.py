"""
一维和多维模型分类
案例： 手写数字识别
"""


import numpy as np
import scipy.io as sio
import scipy.optimize as opt
from sklearn.metrics import classification_report # 这个包是评价报告


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


def load_weight(path):
    data = sio.loadmat(path)
    return data["Theta1"], data["Theta2"]


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cost_function(theta, X, y, λ):  # 正则化的代价函数
    first = y.T @ np.log(sigmoid(X @ theta))
    second = (1 - y.T) @ np.log(1 - sigmoid(X @ theta))
    reg = (λ / (2 * len(X))) * np.sum(np.power(theta[1:], 2))
    # print(first.shape,second.shape,reg)
    return -np.sum(first + second) / len(X) + reg


def gradient_reg(theta, X, y, λ):
    reg = theta[1:] * (λ / len(X))
    reg = np.insert(reg, 0, values=0, axis=0)

    first = (X.T @ (sigmoid(X @ theta) - y)) / len(X)

    return first + reg


def logistic_regression(X, y, λ=1):
    """generalized logistic regression
    args:
        X: feature matrix, (m, n+1) # with incercept x0=1
        y: target vector, (m, )
        l: lambda constant for regularization

    return: trained parameters
    """
    # init theta
    theta = np.zeros(X.shape[1])

    # train it
    res = opt.minimize(fun=cost_function,
                       x0=theta,
                       args=(X, y, λ),
                       method='TNC',
                       jac=gradient_reg,
                       options={'disp': True})
    # get trained parameters
    final_theta = res.x

    return final_theta


def predict(X, theta):
    prob = sigmoid(X @ theta)
    return (prob >= 0.5).astype(int)


if __name__ == "__main__":
    raw_X, raw_y = load_data('ex3data1.mat')  # raw_X--(5000,400),raw_y--(5000,)
    from collections import Counter
    print(Counter(raw_y==10))
    X = np.insert(raw_X, 0, values=np.ones(raw_X.shape[0]), axis=1)  # 插入了第一列（全部为1）,X变为(5000,401)

    # y have 10 categories here. 1..10, they represent digit 0 as category 10 because matlab index start at 1
    # I'll ditit 0, index 0 again
    y_matrix = []
    for k in range(1, 11):
        y_matrix.append((raw_y == k).astype(int))  # 见配图 "向量化标签.png"

    # last one is k==10, it's digit 0, bring it to the first position，最后一列k=10，都是0，把最后一列放到第一列
    print(raw_y,Counter(y_matrix[0]))
    y_matrix = [y_matrix[-1]] + y_matrix[:-1] # 把最后k=10的数组，移到开头第一组的位置
    y = np.array(y_matrix) # y为(10,5000)

    # train k model（训练一维模型）
    theta0_final = logistic_regression(X, y[0])
    print(theta0_final.shape)
    y_pred = predict(X, theta0_final)
    print('Accuracy={}'.format(np.mean(y[0] == y_pred)))

    # train k model（训练k维模型）
    k_theta = np.array([logistic_regression(X, y[k]) for k in range(10)]) # k_theta(10,401)
    print(k_theta.shape)
    y_preds = predict(X, k_theta.T).T # y_reds(10,5000)
    print(y_preds.shape)
    for i in range(10):
        print('Accuracy{}={}'.format(i,np.mean(y[i] == y_preds[i])))

    # 混淆矩阵精度验证
    prob_matrix = sigmoid(X @ k_theta.T) # prob_matrix为(5000,10)的数组
    np.set_printoptions(suppress=True) # 设置打印样式，使输入出结果更美观
    print(prob_matrix.shape)
    y_pre = np.argmax(prob_matrix, axis=1)  # 返回沿轴axis最大值的索引，axis=1代表行--还原回与原始raw_y数据相同的维数，但类别标签不同，一个0-9，一个1-10
    print(y_pre)
    print(np.unique(y_pre))
    # 数据处理，把raw_y数据的标签范围处理成0-9范围内
    y_answer = raw_y.copy()
    y_answer[y_answer == 10] = 0
    print(classification_report(y_answer, y_pre)) # 输出评价报告(包括准确率，召回率，否f1得分等)
