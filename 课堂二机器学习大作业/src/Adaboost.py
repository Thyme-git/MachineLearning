from typing import Union
import numpy as np
from DecisionStump import DecisionStump
from LogisticRegressionClassifier import LogisticRegressionClassifier
import tqdm
import warnings


class Adaboost(object):
    '''
        Adaboost算法实现
    '''
    def __init__(self, base_classifier:str, n_iter:int = 10) -> None:
        '''
            input :
                base_classifier in ['DecisionStump', 'LogisticRegression']
                n_iter : int
        '''
        
        self._base_classifier = base_classifier
        if self._base_classifier == 'DecisionStump':
            self._build_clf = DecisionStump
        elif self._base_classifier == 'LogisticRegression':
            self._build_clf = LogisticRegressionClassifier
        else:
            raise NotImplementedError(f'classifier type {self._base_classifier} not implemented!\n')
        

        self._n_iter = n_iter
        self._alpha  = []
        self._classifier : list[Union[DecisionStump, LogisticRegressionClassifier]] = []

    def fit(self, X:np.ndarray, y:np.ndarray):
        '''
            input :
                X : (b, n)
                y : (b, ) :含有+1、-1两类
                sample_weight : (b, ) X的权重
            return :
                None
        '''
        self._alpha = []
        self._classifier = []

        sample_weight = np.ones(X.shape[0]) / X.shape[0]            #? 初始化权重
        for i in tqdm.tqdm(range(self._n_iter)):
            clf = self._build_clf()
            y_pred = clf.fit(X, y, sample_weight, return_pred=True) #? 以样本权重 D_t 训练 h_t
            err = clf.error_rate                                    #? 计算训练集的加权错误率
            if err > 0.5:
                warnings.warn("\n\033[33mAdaboost quit early because error_rate > 0.5\033[0m", RuntimeWarning)
                break
            alpha = 0.5 * np.log((1-err) / (err + 1e-19))
            sample_weight = sample_weight * np.exp(-1.0 * alpha * y * y_pred)
            sample_weight /= sample_weight.sum()                    #? 权重归一化
            
            self._alpha.append(alpha)
            self._classifier.append(clf)                            #? 保存权重

    def predict(self, X:np.ndarray, max_iter = None):
        '''
            input :
                X : (b, n)
            return :
                y : (b ) :含有+1、-1两类
        '''
        if max_iter is None:
            max_iter = len(self._alpha)

        y_pred = np.zeros(X.shape[0])
        n_iter = min(max_iter, len(self._alpha))
        for i in range(n_iter):
            y_pred += self._alpha[i] * self._classifier[i].predict(X)
        return np.sign(y_pred)


    def save_weight(self, path = '../weight/AdaboostWeight.npy'):
        weight = np.vstack([np.concatenate([clf.weight, np.array([_alpha])])
                            for clf, _alpha in zip(self._classifier, self._alpha) ])
        np.save(path, weight)
    

    def load_weight(self, path = '../weight/AdaboostWeight.npy'):
        weight = np.load(path)
        self._alpha      = []
        self._classifier = []
        for w in weight:
            clf = self._build_clf()
            clf.set_weight(w[:-1])
            self._classifier.append(clf)
            self._alpha.append(w[-1])


if __name__ == '__main__':
    import scienceplots
    from matplotlib import pyplot as plt
    plt.style.use('science')
    from DataReader import DataReader

    dataReader = DataReader()
    accuracy_validate = []
    accuracy_train = []
    n = 20
    adaboost = Adaboost('LogisticRegression', n_iter=n)
    X_train, y_train, index_train,\
    X_validate , y_validate , index_validate = dataReader.get_train_validate_set(np.random.randint(0, 10))
    adaboost.fit(X_train, y_train)
    
    for n_base_classifier in range(1, n+1):
        print(f'use {n_base_classifier} classifiers:')
        
        y_pred = adaboost.predict(X_validate, n_base_classifier)
        acc_validate = (y_pred == y_validate).sum() / len(y_validate)
        accuracy_validate.append(acc_validate)
        
        y_pred = adaboost.predict(X_train, n_base_classifier)
        acc_train = (y_pred == y_train).sum() / len(y_train)
        accuracy_train.append(acc_train)

    plt.plot(np.arange(len(accuracy_train)), np.array(accuracy_train), c = 'r', label = 'Trainiing Set', linewidth = 2)
    plt.plot(np.arange(len(accuracy_validate)), np.array(accuracy_validate), c = 'b', label = 'Validation Set', linewidth = 2)

    plt.legend()
    plt.xlabel('T')
    plt.ylabel('Accuracy')
    # plt.ylabel('Error')
    plt.show()
    plt.close