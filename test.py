import tensorflow as tf
import numpy as np
import copy
from model.utils import *
from model.LSTMlayer import LSTMlayer

def weight_variable(shape):
    initial = tf.truncated_normal(shape=shape, stddev=0.1)
    return tf.Variable(initial_value=initial)

def yield_test():
    for i in range(10):
        yield 1

def for_test():
    for i in range(10):
        a = i
    print(a)
if __name__ == '__main__':
    for_test()
    inputs = tf.constant([[1.0, 2.0, 3.0, 5.0, 6.0],
                        [2.0, 3.0, 4.0, 8.0, 7.0],
                        [3.0, 7.0, 8.0, 0.0, 8.0],
                        [9.0, 8.0, 0.0, 3.0, 1.0]])
    mul = tf.Variable(initial_value=[[2.0, 2.0, 2.0, 2.0, 2.0],
                                     [2.0, 2.0, 2.0, 2.0, 2.0],
                                     [2.0, 2.0, 2.0, 2.0, 2.0],
                                     [2.0, 2.0, 2.0, 2.0, 2.0]], dtype=tf.float32)
    mul = tf.multiply(inputs, mul)
    mul = tf.reduce_sum(mul, axis=-1)
    mul = tf.reduce_sum(mul, axis=-1)
    # x_image = tf.reshape(input, [-1, 4, 5, 1])
    # x_image_shape = tf.shape(x_image)
    # filter_conv = weight_variable([4, 2, 1, 1])
    # filter_conv = tf.constant([[2.0, 2.0],
    #                            [2.0, 2.0],
    #                            [2.0, 2.0],
    #                            [2.0, 2.0]])
    # filter_conv = tf.reshape(filter_conv, [4, 2, 1, 1])
    # conv_result = tf.nn.conv2d(input=x_image, filter=filter_conv, strides=[1, 1, 1, 1], padding='VALID')
    zero_input = tf.zeros(shape=[1, tf.shape(input=inputs)[1]])
    inputs1 = copy.copy(inputs)
    inputs = tf.concat(values=[inputs, zero_input], axis=0)
    test = tf.Variable(
        initial_value=[[1, 2, 3], [4, 5, 6]], dtype=tf.float32
    )

    blank = []
    blank_padding = tf.Variable(
        initial_value=[],
        dtype=tf.float32
    )
    blank_padding1 = tf.Variable(
        initial_value=[],
        dtype=tf.float32
    )
    blank_padding = tf.concat(values=[blank_padding, [23.0]], axis=-1)
    blank_padding = tf.concat(values=[blank_padding, [24.0]], axis=-1)
    blank_padding1 = tf.concat(values=[blank_padding1, [25.0]], axis=-1)
    blank_padding1 = tf.concat(values=[blank_padding1, [26.0]], axis=-1)
    blank.append(blank_padding)
    blank.append(blank_padding1)
    blank = tf.Variable(
        initial_value=blank, dtype=tf.float32
    )

    pri = inputs[0:1, 0:1]
    # =============================================================================
    cnntestinput = [[[1, 2, 3],
                     [2, 3, 4],
                     [4, 5, 6]],
                    [[6, 7, 8],
                     [0, 7, 5],
                     [2, 1, 9]],
                    [[1, 1, 1],
                     [1, 1, 1],
                     [1, 1, 1]],
                    [[2, 2, 2],
                     [2, 2, 2],
                     [2, 2, 2]]]
    cnntestinput = np.array(cnntestinput)
    lstm = LSTMlayer(10)
    cnntestinputtensor = tf.Variable(initial_value=cnntestinput, dtype=tf.float32)
    inputtttt = tf.random_normal(shape=[4, 3, 6], dtype=tf.float32)
    lstmresult, lstmstate = lstm.forward(inputs=cnntestinputtensor)
    # random_tensor = Randomer.random_normal(wshape=[3, 3])
    # cnnadd = cnntestinput + random_tensor
    # # batch.shape = [4, 3, 3] -> [4, 1, 3]
    # cnntestinput_mean = tf.reduce_mean(input_tensor=cnntestinput, axis=1, keep_dims=True)
    # # batch.shape = [4, 1, 3] -> [4, 3, 1]
    # cnntestinput_mean = tf.reshape(tensor=cnntestinput_mean, shape=[-1, 3, 1])
    # filters = [[[1, 2],
    #             [1, 2],
    #             [1, 2]],
    #            [[1, 2],
    #             [1, 2],
    #             [1, 2]]]
    # cnninput = tf.Variable(
    #     initial_value=cnntestinput, dtype=tf.float32
    # )
    # filters = tf.Variable(
    #     initial_value=filters, dtype=tf.float32
    # )
    # conv1d = tf.nn.conv1d(value=cnninput, filters=filters, stride=1, padding='SAME', use_cudnn_on_gpu=True)
    #
    # # =============================================================================
    # test_embedding = np.zeros(shape=[3+1, 10], dtype=np.float32)
    # test_vocab1 = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
    # test_embedding[1] = [float(w) for w in test_vocab1]
    # print(test_embedding)
    #
    # # =============================================================================
    # embedding_number = [[1, 1, 1, 1, 1],
    #                     [2, 2, 2, 2, 2],
    #                     [3, 3, 3, 3, 3],
    #                     [4, 4, 4, 4, 4],
    #                     [5, 5, 5, 5, 5]]
    # embedding_data = tf.get_variable(shape=[5, 5], dtype=tf.float32, name="emb")
    # test_input1 = [[[0, 1, 0, 1],
    #                [1, 2, 1, 2]],
    #               [[2, 3, 2, 3],
    #                [3, 4, 3, 4]]]
    # test_input2 = [[0, 1, 0, 1],
    #                [1, 2, 1, 2]]
    # test_input_embedded = tf.nn.embedding_lookup(embedding_data, test_input1)
    # print(test_input_embedded)
    #
    # # ============================================================================
    #
    # for j in yield_test():
    #     print(j)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(cnntestinputtensor))
        print(sess.run(lstmresult))
        # print(sess.run(random_tensor))
        # print(sess.run(cnnadd))
        # print(sess.run(inputs))
        # print(sess.run(conv1d))
        # print(sess.run(test_input_embedded))
        # print(sess.run(cnntestinput_mean))
        # print(sess.run(random_tensor))