#! encoding=utf8
import os
import pandas as pd
from sklearn.linear_model import Perceptron


if os.path.exists('./step2/result.csv'):
    os.remove('./step2/result.csv')

#********* Begin *********#

train_x = pd.read_csv('./step2/train_data.csv')
train_y = pd.read_csv('./step2/train_label.csv')['target']

test_x = pd.read_csv('./step2/test_data.csv')

net = Perceptron()
net.fit(train_x, train_y)
pred = net.predict(test_x)

pred = pd.DataFrame(pred, names = ['result'])
pred.to_csv('./step2/result.csv', index = False)
#********* End *********#