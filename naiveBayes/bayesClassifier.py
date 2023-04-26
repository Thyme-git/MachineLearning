import numpy as np


class NaiveBayesClassifier(object):
    def __init__(self):
        '''
        self.label_prob表示每种类别在数据中出现的概率
        例如，{0:0.333, 1:0.667}表示数据中类别0出现的概率为0.333，类别1的概率为0.667
        '''
        self.label_prob = {}
        '''
        self.condition_prob表示每种类别确定的条件下各个特征出现的概率
        例如训练数据集中的特征为 [[2, 1, 1],1
                              [1, 2, 2],0
                              [2, 2, 2],1
                              [2, 1, 2],0
                              [1, 2, 3]]1
        标签为[1, 0, 1, 0, 1]
        那么当标签为0时第0列的值为1的概率为0.5，值为2的概率为0.5;
        当标签为0时第1列的值为1的概率为0.5，值为2的概率为0.5;
        当标签为0时第2列的值为1的概率为0，值为2的概率为1，值为3的概率为0;
        当标签为1时第0列的值为1的概率为0.333，值为2的概率为0.666;
        当标签为1时第1列的值为1的概率为0.333，值为2的概率为0.666;
        当标签为1时第2列的值为1的概率为0.333，值为2的概率为0.333,值为3的概率为0.333;
        因此self.label_prob的值如下：     
        {
            0:{
                0:{
                    1:0.5
                    2:0.5
                }
                1:{
                    1:0.5
                    2:0.5
                }
                2:{
                    1:0
                    2:1
                    3:0
                }
            }
            1:
            {
                0:{
                    1:0.333
                    2:0.666
                }
                1:{
                    1:0.333
                    2:0.666
                }
                2:{
                    1:0.333
                    2:0.333
                    3:0.333
                }
            }
        }
        '''
        self.condition_prob = {}
    def fit(self, feature, label):
        '''
        对模型进行训练，需要将各种概率分别保存在self.label_prob和self.condition_prob中
        :param feature: 训练数据集所有特征组成的ndarray
        :param label:训练数据集中所有标签组成的ndarray
        :return: 无返回
        '''


        #********* Begin *********#
        for l in label:
            if l not in self.label_prob.keys():
                self.label_prob[l] = 1
            else:
                self.label_prob[l] += 1
        
        for value in self.label_prob.values():
            value /= len(label)
        
        for feat, l in zip(feature, label):
            if l not in self.condition_prob.keys():
                self.condition_prob[l] = {index:{} for index in range(len(feat))} #优雅的列表推导式
            
            for index, f in enumerate(feat):
                if f not in self.condition_prob[l][index].keys():
                    self.condition_prob[l][index][f] = 1
                else:
                    self.condition_prob[l][index][f] += 1
        
        for l in label:
            for index in range(feature.shape[1]):
                for value in self.condition_prob[l][index].values():
                    value /= len(self.condition_prob[l][index])
        #********* End *********#


    def predict(self, feature):
        '''
        对数据进行预测，返回预测结果
        :param feature:测试数据集所有特征组成的ndarray
        :return:
        '''
        # ********* Begin *********#
        probs = np.zeros((feature.shape[0], len(self.condition_prob)))
        for i, feat in enumerate(feature):
            for label in range(probs.shape[1]):
                probs[i][label] = self.label_prob[label]
                for index, f in enumerate(feat):
                    if f not in self.condition_prob[label][index].keys():
                        probs[i][label] = 0
                        break    
                    probs[i][label] *= self.condition_prob[label][index][f]
        
        return probs.argmax(-1)
        #********* End *********#