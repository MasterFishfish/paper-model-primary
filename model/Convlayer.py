import tensorflow as tf
import numpy as np
import copy
from utils import createFilters

# 共享参数型的conv_layer
class ConvLayer():
    def __init__(self, args):
        self.args = args
        self.paddings = args.paddings
        # 在这个模型里面strides只能为 1
        # 卷积核的窗口大小只能为 2
        self.strides = 1

        self.kernel_size = args.kernel_size
        self.out_channels = args.out_channels

    def forward(self, inputs, filters):
        batch_size = tf.shape(inputs)[0]
        steps = tf.shape(inputs)[1]
        edims = tf.shape(inputs)[2]
        batch_output= []
        for i in range(batch_size):
            sentence = copy.copy(inputs[i])
            # 处理将要卷积的句子的paddings,
            # 理论上sentence是一个tensor
            zero_paddings = tf.zeros(shape=[self.kernel_size-1, edims])
            sentence = tf.concat(values=[sentence, zero_paddings], axis=0)
            seq_outputs = []
            # 计算所有的卷积核在整个句子上得到的结果
            for j in self.out_channels:
                word_output = []
                # 计算某个卷积核在整个句子上的得到的结果
                for loc in range(0, steps, self.strides):
                    # 计算出某一个卷积核在每一步里面得出的结果
                    result = tf.multiply(
                        # 注意[:]切片是左闭右开
                        sentence[i, loc:loc + self.kernel_size] * filters[j]
                    )
                    result = tf.reduce_sum(result, axis=-1)
                    result = tf.reduce_sum(result, axis=-1)
                    word_output.append(result)
                seq_outputs.append(word_output)
            seq_outputs = np.array(seq_outputs)
            seq_outputs = seq_outputs.T
            batch_output.append(seq_outputs)
        batch_output = np.array(batch_output)
        return batch_output

# 非共享参数型
class CnnLayer():
    def __init__(self, in_channels, out_channels, kernel_size=2, strides=1, paddings=1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.strides = strides
        self.kernel_size = kernel_size
        self.paddings = paddings

    def __str__(self):
        str = "this layer has " + self.out_channels + " filters\n" + \
              "every filters has " + self.in_channels + " dimentions same as its inputs\n" + \
              "every filter strides is " + self.strides + " \n" +\
              "the window_size of each filter is" + self.kernel_size

    def forward(self, inputs):
        conv1d = tf.nn.conv1d(
            value=inputs,
            filters=self.out_channels,
            stride=1,
            padding='SAME',
            use_cudnn_on_gpu=True
        )
        return conv1d
