import tensorflow as tf
import copy
import numpy as np
from sklearn.metrics import f1_score
# in_channels 为嵌入词向量的维度
# out_channels 为输出词向量的维度
# kernel_size 为卷积核的第一维的大小 卷积核的大小此处必须为2
# kenel_size * in_channels
# conv1d 输入的filters格式形式[kernel_size, in_channels, out_channels]
def createFilters(in_channels, out_channels, kernel_size=2):
    filters = np.random.rand(kernel_size, in_channels, out_channels)

    for i in range(kernel_size):
        conv_kernel = np.random.randn()
        filters[i] = conv_kernel

    filters_variable = tf.Variable(initial_value=filters, dtype=tf.float32, trainable=True)
    return filters_variable


class Randomer(object):
    stddev = None

    @staticmethod
    def random_normal(wshape):
        return tf.random_normal(wshape, stddev=1.0)

    @staticmethod
    def set_stddev(sd):
        Randomer.stddev = sd

# 获取每一层卷积层结果的某一个时间步骤的词语
# inputs 是一个 list, 元素是每一层的output batch
# inputs.shape = [layernum, batch_size, steps, edims]
# outputs.shape = [batchsize, convlayer_num, edims]
def getTimestepsWord(batch_size, seq_length, convlayer_num, inputs):
    batchlist = copy.copy(inputs)
    for j in range(seq_length):
        all_words_in_the_step = []
        for i in range(batch_size):
            step_words_of_the_seq = []
            for k in range(convlayer_num):
                sentence = batchlist[k][i]
                theword = sentence[j]
                step_words_of_the_seq.append(theword)
            all_words_in_the_step.append(step_words_of_the_seq)
        yield all_words_in_the_step


def getTimeStepsWord_tensor(inputs, batch_size, seq_length, word_dim, convlayer_num=4):
    conv_batches = copy.copy(inputs)

    for j in range(seq_length):
        for i in range(batch_size):
            for k in range(convlayer_num):
                the_batch_per_layer = conv_batches[k]
                if k == 0:
                    the_word_rep = copy.copy(the_batch_per_layer[i:i+1, j:j+1])
                else:
                    # the_word_per_sent.shape = [1, dim_w]
                    the_word_per_sent = the_batch_per_layer[i:i+1, j:j+1]
                    all_layer_word = tf.concat([the_word_rep, the_word_per_sent], axis=0)
            if i == 0:
                the_word_batch = all_layer_word
            else:
                the_word_batch = tf.concat([the_word_batch, all_layer_word], axis=0)
        the_word_batch = tf.reshape(tensor=the_word_batch, shape=[batch_size, convlayer_num, word_dim])
        yield the_word_batch

def evaluate(pred, gold):
    pred_count = np.zeros(3, dtype='int32')
    gold_count = np.zeros(3, dtype='int32')
    hit_count = np.zeros(3, dtype='int32')

    # number of testing documents
    n_test = len(gold)
    error_cases = {}
    for i in range(n_test):
        y_p = int(pred[i])
        y_g = gold[i]
        # print('y_p=', y_p)
        pred_count[y_p] += 1
        gold_count[y_g] += 1
        if y_p == y_g:
            hit_count[y_p] += 1
        else:
            error_cases[i] = [y_p, y_g]

    # number of true predictions
    total_hit = sum(hit_count)
    # accuracy
    acc = float(total_hit) / n_test
    macro_f = f1_score(y_true=gold, y_pred=pred, labels=[0, 1, 2], average='macro')
    return acc, macro_f
