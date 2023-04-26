# Adaboost 算法实现

## 文件目录结构

```
.
├── data
│   ├── data.csv        // 特征数据放这里
│   ├── evaluate.py
│   ├── experiments     // 验证集以及测试集结果在这里
│   └── targets.csv     // 标签放这里
├── src
│   ├── Adaboost.py
│   ├── DataReader.py
│   ├── DecisionStump.py
│   ├── LogisticRegressionClassifier.py
│   └── main.py
├── weight
│   └── AdaboostWeight.npy  // 权重保存
```
没有给出相应的命令行参数时，相关文件保存的位置与上述目录一致

## 测试方法

```
cd src/
python main.py <args>
```

* 使用 ```-t {train, predict}``` 指定训练或者预测
* 使用 ```-c {DecisionStump, LogisticRegression}``` 指定分类器，默认前者
* 使用 ```-d <data file>``` 指定输入的特征(不包含label)
* 使用 ```-l <label file>``` 指定输入的label
* 使用 ```-o <output dir>``` 指定输出的文件夹
* 预测文件形式 ```'/base%d_pred.csv' % (n_base_classifier)```
* 使用 ```-f <weight file>``` 指定权重保存位置

## 测试例子

* 训练
```
cd src/
python main.py -t train -c DecisionStump -d <训练集特征文件> -l <训练集标签文件> // 文件位置可以省略，使用默认值
```

* 测试
```
cd src/
python main.py -t predict -c DecisionStump -d <新的特征文件> -l <新的标签文件>
```