import pandas as pd
import numpy as np

class DataReader(object):
    '''
        获取数据，打乱，划分验证机，训练集
    '''
    def __init__(
            self,
            data_path   : str  = '../data/data.csv',
            target_path : str  = '../data/targets.csv',
            num_fold    : int  = 10,
            train       : bool = True
        ) -> None:
        '''
            input :
                data_path : path for data (only X, label not included)
                            X : (batch_size, n_feat)
                target_path : path for label (y)
                            y : (batch_size, ) containing 0 & 1
        '''
        self._train = train
        
        if train:
            X = pd.read_csv(data_path, header=None)
            y = pd.read_csv(target_path, header=None)

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
        
        elif not train:
            X = pd.read_csv(data_path, header=None)
            self._X = np.array(X.values)


    def get_train_validate_set(self, fold:int):
        '''
            input : 
                fold : 第fold个fold, in range(0, self.numfold)
            return :
                X_train, y_train, index_train, X_validate, y_validate, index_validate
        '''
        if not self._train:
            RuntimeError('Can\'t be used in prediction mode\n')

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
                X_test
        '''
        return self._X
