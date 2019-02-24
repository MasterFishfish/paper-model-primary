import tensorflow as tf
from utils import Randomer

class Linearlayer():
    def __init__(self,
                 w_shape,
                 bias_shape,
                 stddev=None,
                 params=None,
                 active="sigmoid"):
        # w_shape 形状
        # [ widths的大小, heights的大小 ]
        self.weights_shape = w_shape
        self.bias_shape = bias_shape
        if params is None:
            self.wline = tf.Variable(
                initial_value=Randomer.random_normal(self.weights_shape),
                trainable=True
            )
            self.bias = tf.Variable(
                initial_value=Randomer.random_normal(self.bias_shape),
                trainable=True
            )
        else:
            self.wline = params["wline"]
        self.active = active

    # inputs.shape = [batch_size, steps, edims]
    # res.shape = [batch_size, steps, edims]
    def forward(self, inputs):
        w_shp0 = tf.shape(self.wline)[0]
        w_shp1 = tf.shape(self.wline)[1]
        batch_size = tf.shape(inputs)[0]
        # w_line_3dim.shape = [batch_size, edim, edim]
        w_line_3dim = tf.reshape(
            tf.tile(self.wline, [batch_size, 1]),
            [batch_size, w_shp0, w_shp1]
        )
        # linear translate
        res = tf.matmul(inputs, w_line_3dim) + self.bias
        return res


class Linearlayer_3para():
    def __init__(self):
        pass

    def forward(self):
        pass