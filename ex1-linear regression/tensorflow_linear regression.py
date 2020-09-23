"""
基于tensorflow的线性回归
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf


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


def linear_regression(X_data, y_data, alpha, epoch,
                      optimizer=tf.train.GradientDescentOptimizer):  # 这个函数是旧金山的一个大神Lucas Shen写的
    # placeholder for graph input
    X = tf.placeholder(tf.float32, shape=X_data.shape)
    y = tf.placeholder(tf.float32, shape=y_data.shape)

    # construct the graph
    with tf.variable_scope('linear-regression'):
        W = tf.get_variable("weights",
                            (X_data.shape[1], 1),
                            initializer=tf.constant_initializer())  # n*1

        y_pred = tf.matmul(X, W)  # m*n @ n*1 -> m*1

        loss = 1 / (2 * len(X_data)) * tf.matmul((y_pred - y), (y_pred - y), transpose_a=True)  # (m*1).T @ m*1 = 1*1

    opt = optimizer(learning_rate=alpha)
    opt_operation = opt.minimize(loss)

    # run the session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        loss_data = []

        for i in range(epoch):
            _, loss_val, W_val = sess.run([opt_operation, loss, W], feed_dict={X: X_data, y: y_data})
            loss_data.append(loss_val[0, 0])  # because every loss_val is 1*1 ndarray

            if len(loss_data) > 1 and np.abs(
                    loss_data[-1] - loss_data[-2]) < 10 ** -9:  # early break when it's converged
                # print('Converged at epoch {}'.format(i))
                break

    # clear the graph
    tf.reset_default_graph()
    return {'loss': loss_data, 'parameters': W_val}  # just want to return in row vector format


if __name__ == "__main__":
    data = pd.read_csv('ex1data1.txt', names=['population', 'profit'])
    X_data = get_X(data)
    print(X_data.shape, type(X_data))

    y_data = get_y(data).reshape(len(X_data), 1)  # special treatment for tensorflow input data
    print(y_data.shape, type(y_data))
    epoch = 2000
    alpha = 0.01
    optimizer_dict = {'GD': tf.train.GradientDescentOptimizer,
                      'Adagrad': tf.train.AdagradOptimizer,
                      'Adam': tf.train.AdamOptimizer,
                      'Ftrl': tf.train.FtrlOptimizer,
                      'RMS': tf.train.RMSPropOptimizer
                      }
    results = []
    for name in optimizer_dict:
        res = linear_regression(X_data, y_data, alpha, epoch, optimizer=optimizer_dict[name])
        res['name'] = name
        results.append(res)
    print(res["parameters"])
    fig, ax = plt.subplots(figsize=(12, 6))

    for res in results:
        loss_data = res['loss']

        #     print('for optimizer {}'.format(res['name']))
        #     print('final parameters\n', res['parameters'])
        #     print('final loss={}\n'.format(loss_data[-1]))
        ax.plot(np.arange(len(loss_data)), loss_data, label=res['name'])

    ax.set_xlabel('epoch', fontsize=18)
    ax.set_ylabel('cost', fontsize=18)
    ax.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
    ax.set_title('different optimizer', fontsize=18)
    plt.show()
