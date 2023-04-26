
import numpy as np

def alcohol_mean(data):
    '''
    返回红酒数据中红酒的酒精平均含量
    :param data: 红酒数据对象
    :return: 酒精平均含量，类型为float
    '''

    #********* Begin *********#
    mean = np.mean(data['data'], axis=0, keepdims= True)
    std = np.std(data['data'], axis = 0, keepdims= True)

    return (data['data'] - mean)/std

    # return np.mean(data['data'][:, 0])
    #********* End **********#
