import numpy as np
from DecisionStump import BaseClassifier

class LogisticRegressionClassifier(BaseClassifier):
    def __init__(self) -> None:
        self._theta = None
        self._err = 1.0
        self._X_mean = 0.0
        self._X_std = 1.0
    
    def _sigmoid(
            self,
            X:np.ndarray
        ):
        '''
            elem-wise sigmoid
            input : 
                X : (X.shape)
            return :
                y : (X.shape)
        '''
        return 1 / (np.exp(-X) + 1)


    def _grad(
            self,
            X:np.ndarray,
            y:np.ndarray,
            sample_weight : np.ndarray = None
        ):
        '''
            input :
                X : (b, n)
                y : (b, ) :含有+1、0两类
                sample_weight : (b, )
            return :
                gradient of self._theta
        '''
        y_delta = self._sigmoid(X.dot(self._theta))-y
        if sample_weight is not None:
            return (sample_weight * y_delta).dot(X)
    
        return y_delta.dot(X) / len(y_delta)
    

    def fit(
            self,
            X:np.ndarray,
            y:np.ndarray,
            sample_weight : np.ndarray = None,
            return_pred = False,
            max_iter = 10000,
            lr = 1e-1,
            garma = 0.99
        ):
        '''
            input :
                X : (b, n)
                y : (b, ) :含有+1、-1两类
                sample_weight : (b, ) X的权重
                return_pred : 是否要返回 X 的预测值 y_pred
            return :
                None / y_pred : 训练集的预测值
        '''
        y[np.argwhere(y == -1.0)] = 0.0

        if sample_weight is None:
            sample_weight = np.ones(X.shape[0]) / X.shape[0]
        
        self._theta = np.ones(X.shape[1]+1)



        self._X_mean = X.mean(axis=0)       
        self._X_std = X.std(axis=0)
        X = (X - self._X_mean) / self._X_std    #? 正则化处理
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        for _ in range(max_iter):
            g = self._grad(X, y, sample_weight) #? 计算梯度
            self._theta -= lr * g
            if _ % 100 == 0:
                lr *= garma

        
        y_pred = self.predict(X, train=True)
        y[np.argwhere(y == 0.0)] = -1.0
        self._err = np.inner((y_pred != y), sample_weight)
        if return_pred:
            return y_pred
    

    def predict(
            self,
            X : np.ndarray,
            train = False
        ):
        '''
            input :
                X : (b, n)
                train :是否在训练
            return :
                y : (b, ) :含有+1、-1两类
        '''
        if not train:           #? 训练过程中会在fit中正则化处理
            X = (X - self._X_mean) / self._X_std
            X = np.hstack([np.ones((X.shape[0], 1)), X])

        y = self._sigmoid(X.dot(self._theta))
        y[np.argwhere(y < 0.5)] = -1.0
        y[np.argwhere(y >= 0.5)] = 1.0
        return y
    
    @property
    def error_rate(self):
        '''
            weighted error rate of training set
            used in adaboost
        '''
        return self._err

    @property
    def weight(self) -> np.ndarray:
        '''
            获取训练权重 以np array形式返回
        '''
        return np.concatenate([self._theta, self._X_mean, self._X_std])


    def save_weight(self, path = '../LogisticRegressionWeight.npy'):
        np.save(path, self.weight)

    
    def set_weight(self, to_weight: np.ndarray):
        '''
            set weight from to_weight
        '''
        n_feat = int(len(to_weight) / 3)
        self._theta  = to_weight[0 : n_feat+1]
        self._X_mean = to_weight[n_feat+1 : 2*n_feat+1]
        self._X_std  = to_weight[2*n_feat+1 : ]


    def load_weight(self, path = '../LogisticRegressionWeight.npy'):
        weight = np.load(path)
        self.set_weight(weight)