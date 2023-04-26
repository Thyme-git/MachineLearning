from sklearn.datasets import load_wine
import numpy as np
wine_dataset = load_wine()

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
answer = scaler.fit_transform(wine_dataset['data'])

import step4

result = step4.scaler(wine_dataset)

diff = np.sum(np.square(result-answer))

if diff < 0.1:
    print('标准化成功')
else:
    print('你的结果与答案的L2误差为%.6f' % diff)
