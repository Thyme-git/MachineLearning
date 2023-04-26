# -*- coding: utf-8 -*-

import numpy as np
import warnings
warnings.filterwarnings("ignore")

def sigmoid(x):
    '''
    sigmoid函数
    :param x: 转换前的输入
    :return: 转换后的概率
    '''
    return 1/(1+np.exp(-x))


def fit(x, y, eta=1e-3, n_iters=10000):
    '''
    训练逻辑回归模型
    :param x: 训练集特征数据，类型为ndarray
    :param y: 训练集标签，类型为ndarray
    :param eta: 学习率，类型为float
    :param n_iters: 训练轮数，类型为int
    :return: 模型参数，类型为ndarray
    '''
    #   请在此添加实现代码   #
    #********** Begin *********#
    b_and_w = np.ones(x.shape[1] + 1)
    x = np.hstack([np.ones((x.shape[0], 1)), x])

    for _ in range(n_iters):
        y_pred = sigmoid((x*b_and_w).sum(axis = 1))
        b_and_w -= eta*(y-y_pred).reshape((-1, 1))*x

    return b_and_w

    #********** End **********#
