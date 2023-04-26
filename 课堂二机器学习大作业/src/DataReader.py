import pandas as pd
import numpy as np

class DataReader(object):
    '''
        获取数据，打乱，划分验证机，训练集
    '''
    def __init__(
            self,
            data_path   : str = '../data/data.csv',
            target_path : str = '../data/targets.csv',
            num_fold    : int = 10
        ) -> None:
        '''
            input :
                data_path : path for data (only X, label not included)
                            X : (batch_size, n_feat)
                target_path : path for label (y)
                            y : (batch_size, ) containing 0 & 1
        '''
        X = pd.read_csv(data_path)
        y = pd.read_csv(target_path)

        X     = np.array(X.values)
        y     = np.array(y.values)
        index = np.arange(start=1, stop=y.shape[0]+1)

        data = np.concatenate([X, y, index.reshape((-1, 1))], axis = 1)
        np.random.shuffle(data)

        self._X     = data[:, :-2]
        self._y     = data[:, -2].reshape(-1)
        self._index = data[:, -1].reshape(-1)

        '''牛马数据集'''
        self._y[np.argwhere(self._y == 0.0)] = -1.0

        self._num_fold = num_fold
        self._size = self._y.shape[0]
        self._step_size = int(self._size / self._num_fold)


    def get_train_validate_set(self, fold:int):
        '''
            input : 
                fold : 第fold个fold, in range(0, self.numfold)
            return :
                X_train, y_train, index_train, X_validate, y_validate, index_validate
        '''
        try:
            assert (fold >= 0) and (fold < self._num_fold)
        except:
            ValueError(f'invalid fold num : {fold}')
        
        range_left  = self._step_size * fold
        range_right = range_left + self._step_size
        X_train     = np.concatenate([self._X[:range_left], self._X[range_right:]], axis=0)
        y_train     = np.concatenate([self._y[:range_left], self._y[range_right:]], axis=0)
        index_train = np.concatenate([self._index[:range_left], self._index[range_right:]], axis=0)
        X_validate      = self._X[range_left:range_right]
        y_validate      = self._y[range_left:range_right]
        index_validate  = self._index[range_left:range_right]

        return X_train, y_train, index_train, X_validate, y_validate, index_validate


    def get_test_set(self):
        '''
            return the whole set as test set
            return:
                X_test, y_test, index_test
        '''
        return self._X, self._y, self._index
    


if __name__ == '__main__':
    
    from sklearn.model_selection import StratifiedKFold
    import scienceplots
    from LogisticRegressionClassifier import LogisticRegressionClassifier
    from matplotlib import pyplot as plt
    plt.style.use('science')
    def visualize_cv(cv, X, y): 
        fig, ax = plt.subplots(figsize=(10, 5)) 
    
        for ii, (tr, tt) in enumerate(cv.split(X, y)): 
            p1 = ax.scatter(tr, [ii] * len(tr), c="orangered", marker="_", lw=8) 
            p2 = ax.scatter(tt, [ii] * len(tt), c="skyblue", marker="_", lw=8) 
            ax.set( 
                title='10-Fold Cross Validation', 
                xlabel="Data Index", 
                ylabel="Fold", 
                ylim=[cv.n_splits, -1], 
            ) 
            ax.legend([p1, p2], ["Validation","Training"]) 
    
        plt.show()

    reader = DataReader()
    # X_train, y_train, index_train,\
    # X_validate , y_validate , index_validate = reader.get_train_validate_set(0)
    # clf = LogisticRegressionClassifier()
    # clf.fit(X_train, y_train)
    # y_pred = clf.predict(X_validate)
    # acc = (y_pred == y_validate).sum() / len(y_validate)
    # print(acc)

    X, y, index = reader.get_test_set()
    cv = StratifiedKFold(n_splits=10, shuffle=True) 
    visualize_cv(cv, X, y)
