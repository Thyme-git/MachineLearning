import os
import argparse
from mindspore import context, set_context, PYNATIVE_MODE

# set_context(mode=PYNATIVE_MODE)

parser = argparse.ArgumentParser(description='MindSpore LeNet Example')
parser.add_argument('--device_target', type=str, default="CPU", choices=['Ascend', 'GPU', 'CPU'])

args = parser.parse_known_args()[0]
# context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)

import os
import requests

requests.packages.urllib3.disable_warnings()

def download_dataset(dataset_url, path):
    filename = dataset_url.split("/")[-1]
    save_path = os.path.join(path, filename)
    if os.path.exists(save_path):
        return
    if not os.path.exists(path):
        os.makedirs(path)
    res = requests.get(dataset_url, stream=True, verify=False)
    with open(save_path, "wb") as f:
        for chunk in res.iter_content(chunk_size=512):
            if chunk:
                f.write(chunk)
    print("The {} file is downloaded and saved in the path {} after processing".format(os.path.basename(dataset_url), path))

train_path = "datasets/MNIST_Data/train"
test_path = "datasets/MNIST_Data/test"

download_dataset("https://mindspore-website.obs.myhuaweicloud.com/notebook/datasets/mnist/train-labels-idx1-ubyte", train_path)
download_dataset("https://mindspore-website.obs.myhuaweicloud.com/notebook/datasets/mnist/train-images-idx3-ubyte", train_path)
download_dataset("https://mindspore-website.obs.myhuaweicloud.com/notebook/datasets/mnist/t10k-labels-idx1-ubyte", test_path)
download_dataset("https://mindspore-website.obs.myhuaweicloud.com/notebook/datasets/mnist/t10k-images-idx3-ubyte", test_path)

import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C
import mindspore.dataset.vision.c_transforms as CV
from mindspore.dataset.vision import Inter
from mindspore import dtype as mstype

def create_dataset(data_path, batch_size=32, repeat_size=1,num_parallel_workers=1):
    # 定义数据集
    mnist_ds = ds.MnistDataset(data_path)
    # Todo 设置放缩的大小
    resize_height, resize_width = 32, 32
    # Todo 归一化
    rescale = 1 / 255
    shift = 0.0
    rescale_nml = 1 / 0.3081
    shift_nml = -1 * 0.1307 / 0.3081

    # 定义所需要操作的map映射
    resize_op = CV.Resize((resize_height, resize_width), interpolation=Inter.LINEAR)
    rescale_nml_op = CV.Rescale(rescale_nml, shift_nml)
    rescale_op = CV.Rescale(rescale, shift)
    hwc2chw_op = CV.HWC2CHW()
    type_cast_op = C.TypeCast(mstype.int32)

    # 使用map映射函数，将数据操作应用到数据集
    mnist_ds = mnist_ds.map(operations=type_cast_op, input_columns="label", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=[resize_op, rescale_op, rescale_nml_op, hwc2chw_op], input_columns="image", num_parallel_workers=num_parallel_workers)

    # Todo 进行shuffle、batch、repeat操作
    buffer_size = 10000
    mnist_ds = mnist_ds.shuffle(buffer_size = buffer_size)
    mnist_ds = mnist_ds.batch(batch_size = batch_size)
    mnist_ds = mnist_ds.repeat(repeat_size)

    return mnist_ds

import mindspore.nn as nn
from mindspore.common.initializer import Normal
import mindspore as ms
import mindspore.ops as ops
import numpy as np


# Todo 根据LeNet5网络结构神经网络
class LeNet5(nn.Cell):
    """
    Lenet网络结构
    """
    def __init__(self, num_class=10, num_channel=1):
        super(LeNet5, self).__init__()
        
        # self.transpose = ops.Transpose()

        # (1, 32, 32)
        self.conv1 = nn.Conv2d(in_channels = num_channel, out_channels = 6, kernel_size = (5, 5), has_bias=True, pad_mode='valid')
        self.relu1 = nn.ReLU()      # (6, 28, 28)
        self.max_pool1 = nn.MaxPool2d(kernel_size = (2, 2), stride = 2) # (6, 14, 14)

        self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size = (5, 5), has_bias=True, pad_mode='valid')
        self.relu2 = nn.ReLU()      # (16, 10, 10)
        self.max_pool2 = nn.MaxPool2d(kernel_size = (2, 2), stride = 2) # (16, 5, 5)

        self.conv3 = nn.Conv2d(in_channels = 16, out_channels = 120, kernel_size = (5, 5), has_bias = True, pad_mode='valid') # (120, 1, 1)
        self.squeeze = ops.Squeeze(axis = (-1, -2))
        self.fc = nn.Dense(120, 84, activation = 'tanh') # (B, 84)

        self.out_fc = nn.Dense(84, num_class, activation = None)
    
    def construct(self, x):
        # 使用定义好的运算构建前向网络
        # (b, c, h, w)

        x = self.relu1(self.conv1(x))
        x = self.max_pool1(x)

        x = self.relu2(self.conv2(x))
        x = self.max_pool2(x)

        x = self.squeeze(self.conv3(x))
        x = self.fc(x)

        x = self.out_fc(x)

        return x

# 实例化网络
net = LeNet5()

# MindSpore支持的损失函数有SoftmaxCrossEntropyWithLogits、L1Loss、MSELoss等。
net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

# MindSpore支持的优化器有Adam、AdamWeightDecay、Momentum等。这里使用Momentum优化器为例。
net_opt = nn.Momentum(net.trainable_params(), learning_rate=0.01, momentum=0.9)

from mindspore.train.callback import ModelCheckpoint, CheckpointConfig

config_ck = CheckpointConfig(save_checkpoint_steps=1875, keep_checkpoint_max=10)

ckpoint = ModelCheckpoint(prefix="checkpoint_lenet", config=config_ck)

from mindspore.nn import Accuracy
from mindspore.train.callback import LossMonitor
from mindspore import Model

def train_net(model, epoch_size, data_path, repeat_size, ckpoint_cb, sink_mode):
    """定义训练的方法"""
    # 加载训练数据集
    ds_train = create_dataset(os.path.join(data_path, "train"), repeat_size = repeat_size)
    model.train(epoch_size, ds_train, callbacks=[ckpoint_cb, LossMonitor(125)], dataset_sink_mode=sink_mode)

def test_net(model, data_path):
    """定义验证的方法"""
    ds_eval = create_dataset(os.path.join(data_path, "test"))
    acc = model.eval(ds_eval, dataset_sink_mode=False)
    print(f"acc : {acc['Accuracy']:.4f}")

train_epoch = 20
mnist_path = "./datasets/MNIST_Data"
dataset_size = 1
model = Model(net, net_loss, net_opt, metrics={"Accuracy": Accuracy()})
train_net(model, train_epoch, mnist_path, dataset_size, ckpoint, False)
test_net(model, mnist_path)

from mindspore import load_checkpoint, load_param_into_net
# 加载已经保存的用于测试的模型
param_dict = load_checkpoint(f"checkpoint_lenet-{train_epoch}_1875.ckpt") # 使用最后一个检查点
# 加载参数到网络中
load_param_into_net(net, param_dict)


import numpy as np
from mindspore import Tensor

# 定义测试数据集，batch_size设置为1，则取出一张图片
ds_test = create_dataset(os.path.join(mnist_path, "test"), batch_size=1).create_dict_iterator()
data = next(ds_test)

# images为测试图片，labels为测试图片的实际分类
images = data["image"].asnumpy()
labels = data["label"].asnumpy()

# 使用函数model.predict预测image对应分类
output = model.predict(Tensor(data['image']))
predicted = np.argmax(output.asnumpy(), axis=1)

# 输出预测分类与实际分类
print(f'Predicted: "{predicted[0]}", Actual: "{labels[0]}"')