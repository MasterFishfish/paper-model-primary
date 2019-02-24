import tensorflow as tf
import numpy as np
import argparse
import time
import math

from model.Convlayer import CnnLayer
from model.utils import getTimestepsWord
from model.Linearlayer import Linearlayer
from model.LSTMlayer import LSTMlayer
from model.CPAN import CPAN
from data_prepare.readfile import load_data14semeval
from data_prepare.readfile import *
# 编写模型运行的代码


# ds_name 使用哪一个dataset
# bs batch_size的大小
# dim_words 嵌入词向量的维度
# dim_positions 整理后的某个位置的维度
# dropout_rate
# rnn_type
# n_epoch
# lr learning_rate 学习率
# dim_h
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TNet settings')
    parser.add_argument("-ds_name", type=str, default="14semeval_rest", help="dataset name")
    # parser.add_argument("-n_filter", type=int, default=50, help="number of convolutional filters")
    parser.add_argument("-bs", type=int, default=64, help="batch size")
    parser.add_argument("-dim_w", type=int, default=300, help="dimension of word embeddings")
    # parser.add_argument("-dim_e", type=int, default=25, help="dimension of episode")
    # parser.add_argument("-dim_func", type=int, default=10, help="dimension of functional embeddings")
    parser.add_argument("-dim_p", type=int, default=30, help="dimension of position embeddings")
    parser.add_argument("-dropout_rate", type=float, default=0.3, help="dropout rate for sentimental features")
    parser.add_argument("-dim_h", type=int, default=50, help="dimension of hidden state")
    parser.add_argument("-rnn_type", type=str, default='LSTM', help="type of recurrent unit")
    parser.add_argument("-n_epoch", type=int, default=100, help="number of training epoch")
    parser.add_argument("-dim_y", type=int, default=3, help="dimension of label space")
    parser.add_argument("-lr", type=float, default=0.002, help="learning rate")
    parser.add_argument("-grad_clip", type=int, default=5, help="grad clip")
    # parser.add_argument("-lambda", type=float, default=1e-4, help="L2 coefficient")
    parser.add_argument("-did", type=int, default=2, help="gpu device id")
    # parser.add_argument("-connection_type", type=str, default="AS", help="connection type, only AS and LF are valid")

    args = parser.parse_args()

    if "14" in args.ds_name:
        datasets, num_datasets, vocab, embeddings, seq_len = load_data14semeval(dataset_name=args.ds_name,
                                                                                dim_w=args.dim_w,
                                                                                batch_size=args.bs)
    elif "15" in args.ds_name:
        pass

    elif "16" in args.ds_name:
        pass

    # update embedding dims
    args.dim_w = len(embeddings[1])
    print(args)
    args.embeddings = embeddings

    # print seqence steps length
    print("data_sent_check:", (datasets[0][0]["sent_wrods_id"] == seq_len[0]))
    print("data_aspect_check:", (datasets[0][0]["target_words_id"] == seq_len[1]))
    args.sent_len = len(datasets[0][0]["sent_words_id"])
    args.target_len = len(datasets[0][0]["target_words_id"])
    print("sentence length is: ", args.sent_len)
    print("target length is: ", args.target_len)
    print("length of padded training set:", len(datasets[0]))

    n_train = num_datasets[0]
    n_test = num_datasets[1]
    n_train_batches = math.ceil(n_train / args.bs)
    print("n batches of training set:", n_train_batches)
    n_test_batches = math.ceil(n_train / args.bs)
    print("n batches of test set:", n_test_batches)


    train_set = datasets[0]
    test_set = datasets[1]
    model = CPAN(args=args)
    result_strings = model.get_modelresult(epoch=args.n_epoch,
                                           n_train_batches=n_train_batches,
                                           n_test_batches=n_test_batches,
                                           train_set=train_set,
                                           test_set=test_set,
                                           embeddings=args.embeddings)

    cur_model_name = "CPAN"

    result_logs = ['-------------------------------------------------------\n']
    params_string = str(args)
    #result_logs.append("Running model: %s\n" % cur_model_name)
    result_logs.append(params_string + "\n")
    result_logs.extend(result_strings)
    result_logs.append('-------------------------------------------------------\n')
    if not os.path.exists('./log'):
        os.mkdir('log')
    with open("./log/original_%s_%s.txt" % (cur_model_name, args.ds_name), 'a') as fp:
        fp.writelines(result_logs)


