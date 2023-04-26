#encoding=utf8
import numpy as np
#构建感知机算法
class Perceptron(object):
    def __init__(self, learning_rate = 0.01, max_iter = 200):
        self.lr = learning_rate
        self.max_iter = max_iter
    
    def fit(self, data, label):
        '''
        input:data(ndarray):训练数据特征
              label(ndarray):训练数据标签
        output:w(ndarray):训练好的权重
               b(ndarry):训练好的偏置
        '''
        #编写感知机训练方法，w为权重，b为偏置
        self.w = np.array([1.]*data.shape[1])
        self.b = np.array([1.])
        #********* Begin *********#
        done = True
        for _ in range(self.max_iter):
            done = True
            for (x, y) in zip(data, label):
                if y*((self.w*x).sum() + self.b) <= 0:
                    done = False
                    self.w += self.lr*y*x
                    self.b += self.lr*y
            if done:
                break
        #********* End *********#
    
    def predict(self, data):
        '''
        input:data(ndarray):测试数据特征
        output:predict(ndarray):预测标签
        '''
        #********* Begin *********#
        predict = np.sign((self.w*data).sum(axis = 1) + self.b)
        #********* End *********#
        return predict