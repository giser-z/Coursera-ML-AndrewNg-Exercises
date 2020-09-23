"""
垃圾邮件分类---预处理
"""

'''
预处理主要包括以下8个部分：
  1. 将大小写统一成小写字母；
  2. 移除所有HTML标签，只保留内容。
  3. 将所有的网址替换为字符串 “httpaddr”.
  4. 将所有的邮箱地址替换为 “emailaddr”
  5. 将所有dollar符号($)替换为“dollar”.
  6. 将所有数字替换为“number”
  7. 将所有单词还原为词源，词干提取
  8. 移除所有非文字类型
  9.去除空字符串‘’
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn import svm
import nltk.stem as ns
import re


def preprocessing(email):
    # 1. 统一成小写
    email = email.lower()

    # 2. 去除html标签
    email = re.sub('<[^<>]>', ' ', email)

    # 3. 将网址替换为字符串 “httpaddr”.
    email = re.sub('(http|https)://[^\s]*', 'httpaddr', email)

    # 4. 将邮箱地址替换为 “emailaddr”
    email = re.sub('[^\s]+@[^\s]+', 'emailaddr', email)

    # 5.所有dollar符号($)替换为“dollar”.
    email = re.sub('[\$]+', 'dollar', email)

    # 6.匹配数字，将数字替换为“number”
    email = re.sub('[0-9]+', 'number', email)  # 匹配一个数字， 相当于 [0-9]，+ 匹配1到多次

    # 7. 词干提取
    tokens = re.split('[ \@\$\/\#\.\-\:\&\*\+\=\[\]\?\!\(\)\{\}\,\'\"\>\_\<\;\%]', email)
    tokenlist = []

    s = ns.SnowballStemmer('english')

    for token in tokens:

        # 8. 移除非文字类型
        email = re.sub('[^a-zA-Z0-9]', '', email)
        stemmed = s.stem(token)

        # 9.去除空字符串‘’
        if not len(token): continue
        tokenlist.append(stemmed)

    return tokenlist


def email2VocabIndices(email, vocab):
    """提取存在单词的索引"""
    token = preprocessing(email)
    print(token)
    index = [i for i in range(len(token)) if token[i] in vocab]
    return index


def email2FeatureVector(email):
    """
    将email转化为词向量，n是vocab的长度。存在单词的相应位置的值置为1，其余为0
    """
    df = pd.read_table('data/vocab.txt', names=['words'])
    vocab = df.values  # return array
    vector = np.zeros(len(vocab))  # init vector
    vocab_indices = email2VocabIndices(email, vocab)
    print(vocab_indices)  # 返回含有单词的索引
    # 将有单词的索引置为1
    for i in vocab_indices:
        vector[i] = 1
    return vector


if __name__ == '__main__':
    with open("data/emailSample1.txt") as file:
        sample_email = file.read()
        print(sample_email)
    email = preprocessing(sample_email)
    vector = email2FeatureVector(sample_email)
    print('length of vector = {}\nnum of non-zero = {}'.format(len(vector), int(vector.sum())))
    print(vector.shape,vector)
