import numpy as np

class BaseClassifier(object):
    '''分类器的基类'''

    def __init__(self, *args) -> None:
        '''初始化权重'''
        self._err = 1.0
        pass

    def fit(self, X, y, sample_weight, *args):
        '''以输入的数据训练分类器'''
        raise NotImplementedError
    
    def predict(self, X, *args) -> np.ndarray:
        '''输出X的预测结果'''
        raise NotImplementedError
    
    @property
    def error_rate(self):
        '''返回加权错误率'''
        return self._err
    
    @property
    def weight(self) -> np.ndarray:
        '''返回分类器模型的权重'''
        raise NotImplementedError

class DecisionStump(BaseClassifier):
    '''决策树桩实现'''

    def __init__(self) -> None:
        self._thredhold              = None
        self._compare_method         = None
        self._critical_dimension     = None
        self._min_err                = 1.0
    
    def fit(
            self,
            X:np.ndarray,
            y:np.ndarray,
            sample_weight : np.ndarray = None,
            return_pred = False
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
        # self.batch_size = X.shape[0]
        self._min_err = 1.0
        if sample_weight is None:
            sample_weight = np.ones(X.shape[0]) / X.shape[0]
        
        num_steps = 1000
        for dim in range(X.shape[1]):
            dim_min = np.min(X[:, dim])
            step_size = (np.max(X[:, dim]) - dim_min) / num_steps

            for step in range(num_steps + 1):
                thredhold = dim_min + step * step_size              #? 当前遍历的遍历的阈值
                for compare_method in [0, 1]:
                    if compare_method == 0:
                        y_pred = self.predict(X, compare_method, thredhold, dim)
                    else:
                        y_pred = - y_pred
                    err = np.inner((y_pred != y), sample_weight)    #? 计算加权误差
                    if err < self._min_err:                         #? 权重保存
                        self._min_err = err
                        self._thredhold = thredhold
                        self._compare_method = compare_method
                        self._critical_dimension = dim

        if return_pred:
            return self.predict(X, self._compare_method, self._thredhold, self._critical_dimension)


    def predict(
            self,
            X:np.ndarray,
            compare_method = None,
            thredhold = None,
            dim = None
        ):
        '''
            input :
                X : (b, n)
            return :
                y : (b ) :含有+1、-1两类
        '''
        try:
            assert self._thredhold is not None or thredhold is not None
        except:
            ReferenceError("predict method must be referred after fit")

        
        if compare_method is None:
            compare_method = self._compare_method
        
        if thredhold is None:
            thredhold = self._thredhold
        
        if dim is None:
            dim = self._critical_dimension
        
        y = np.ones(X.shape[0])
        if compare_method == 0:
            y[np.argwhere(X[:, dim] < thredhold)] = -1.0
        else:
            y[np.argwhere(X[:, dim] >= thredhold)] = -1.0
        return y


    @property
    def error_rate(self):
        '''
            weighted error rate of training set
            used in adaboost
        '''
        return self._min_err
    
    @property
    def weight(self) -> np.ndarray:
        '''
            获取训练权重 以np array形式返回
        '''
        return np.array([self._thredhold, self._compare_method, self._critical_dimension])


    def save_weight(self, path = '../DecisionStumpWeight.npy'):
        np.save(path, self.weight)

    
    def set_weight(self, to_weight: np.ndarray):
        '''
            set weight from to_weight
        '''
        self._thredhold          = to_weight[0]
        self._compare_method     = int(to_weight[1])
        self._critical_dimension = int(to_weight[2])

    def load_weight(self, path = '../DecisionStumpWeight.npy'):
        weight = np.load(path)
        self.set_weight(weight)