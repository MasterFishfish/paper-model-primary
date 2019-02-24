# Convolution Phrase-Attention Network
import tensorflow as tf
from LSTMlayer import LSTMlayer
from Convlayer import CnnLayer
from utils import *
from Dotattention import DotAttentionLayer
from Linearlayer import Linearlayer
import pandas as pd
import time

def batch_generate(datasets, batch_size, idx):
    batch_set = datasets[idx*batch_size:(idx+1)*batch_size]
    batch_data = pd.DataFrame.from_dict(data=batch_set)
    target_fields = ['sent_word_id', 'target_word_id', 'y']
    batch_input_var = []
    for key in target_fields:
        # print("for %s data" % key)
        data = list(batch_data[key].values)
        # print(data[0])
        # print(data[5])
        if key == 'pw':
            # for position weights
            # batch_input_var.append(theano.shared(value=np.array(data, dtype='float32')))
            batch_input_var.append(np.array(data, dtype='float32'))
        else:
            # batch_input_var.append(theano.shared(value=np.array(data, dtype='int32')))
            batch_input_var.append(np.array(data, dtype='int32'))
    return batch_input_var

class CPAN():

    def __init__(self, args, vocab):
        # batch_size
        self.batch_size = args.bs
        # dim of the position words
        self.brnn_out = args.dim_p
        # the dropcell rate
        self.dropout_rate = args.dropout_rate
        # the word embeddings
        # self.embeddings_weights = args.embeddings
        # the type of Bidirectional neural network
        self.rnn_type = args.rnn_type
        # the number of epoch
        self.n_epoch = args.n_epoch
        # learning_rate of the model
        self.learn_rate = args.learn_rate
        # dimension of label space (0:negative, 1:positive, 2:neutral)
        self.n_label = args.dim_y
        # steps of aspects
        self.aspects_len = args.target_len
        # steps of sentence
        self.sent_len = args.sent_len
        # dimension of word embeddings
        self.dim_w = args.dim_w
        # vocab
        self.vocab = vocab
        # the kinds of labels
        self.grad_clip = args.grad_clip

        # 运行模型
        self.build_inputs()
        self.build_model()
        self.build_loss()
        self.build_optimizer()

    def build_inputs(self):
        # 模型的输入
        with tf.name_scope("inputs"):
            self.inputs = tf.placeholder(dtype=tf.int32,
                                         shape=[self.batch_size, self.sent_len, self.dim_w],
                                         name="inputs_sentence")

            # self.reverse_input = tf.placeholder(dtype=tf.int32,
            #                                     shape=[self.batch_size, self.sent_len, self.dim_w],
            #                                     name="reverse_input")

            self.aspects = tf.placeholder(dtype=tf.int32,
                                          shape=[self.batch_size, self.aspects_len, self.dim_w],
                                          name="aspects")

            # self.reverse_aspects = tf.placeholder(dtype=tf.int32,
            #                                       shape=[self.batch_size, self.aspects_len, self.dim_w])

            # 注意 self.y 的形状和dtype
            self.y = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.n_label], name="labels_dim")

            # embeddings
            self.embeddings = tf.placeholder(dtype=tf.float32,
                                        shape=[len(self.vocab), self.dim_w],
                                        name="embeddings")

            self.lstm_inputs = tf.nn.embedding_lookup(params=self.embeddings,
                                                      ids=self.inputs)

            # self.reverse_lstm_inputs = tf.nn.embedding_lookup(params=self.embeddings,
            #                                                   ids=self.reverse_input)
            self.reverse_lstm_inputs = tf.reverse(tensor=self.lstm_inputs,
                                                  axis=[1],
                                                  name="reverse_sent")

            self.aspects_inputs = tf.nn.embedding_lookup(params=self.embeddings,
                                                         ids=self.aspects)

            # self.reverse_aspects_inputs = tf.nn.embedding_lookup(params=self.embeddings,
            #                                                      ids=self.aspects)
            self.reverse_aspects_inputs = tf.reverse(tensor=self.aspects_inputs,
                                                     axis=[1],
                                                     name="reverse_aspects")

    def build_model(self):
        # 计算前向传播神经网络和后向传播神经网络的连结
        with tf.name_scope("Bidirectional_lstm"):
            # dropout wapper ?
            single_lstm = LSTMlayer(hidden_size=self.brnn_out)
            single_lstm_reverse = LSTMlayer(hidden_size=self.brnn_out)
            single_lstm_aspects = LSTMlayer(hidden_size=self.brnn_out)
            single_lstm_aspects_reverse = LSTMlayer(hidden_size=self.brnn_out)

        # ctx 的双向传递神经网络计算
        last_state, step_rep = single_lstm.forward(inputs=self.lstm_inputs)

        reverse_last_state, reverse_step_rep = single_lstm_reverse.forward(inputs=self.reverse_lstm_inputs)

        # aspects 的双向传递神经网络计算
        aspects_last_state, aspects_step_rep = single_lstm_aspects.forward(inputs=self.aspects_inputs)

        reverse_aspects_state, reverse_aspects_rep = \
            single_lstm_aspects_reverse.forward(inputs=self.reverse_aspects_inputs)

        ### 连结生成word_representation
        ### concat之后生成的rep.dim = 2 * brnn_out
        sentence_rep = tf.concat(values=[step_rep, reverse_step_rep], axis=-1)
        conv1_input_wordim = tf.shape(sentence_rep)[-1]
        aspects_rep = tf.concat(values=[aspects_step_rep, reverse_aspects_rep], axis=-1)
        aspects_input_dim = tf.shape(aspects_rep)[-1]

        ### 生成 filters 在后面的layer中共享参数和变量
        filters = createFilters(in_channels=conv1_input_wordim, out_channels=conv1_input_wordim)

        # 执行多层卷积
        # 全部要改，conv1d 不可以共享参数
        with tf.name_scope("conv1d_layers"):
            ### 第一次卷积，两词短语
            convlayer_1 = CnnLayer(in_channels=conv1_input_wordim, out_channels=conv1_input_wordim)
            conv1_result = convlayer_1.forward(inputs=sentence_rep)

            ### 第二次卷积，三词短语
            convlayer_2 = CnnLayer(in_channels=conv1_input_wordim, out_channels=conv1_input_wordim)
            conv2_result = convlayer_2.forward(inputs=conv1_result)

            ### 第三次卷积， 四词短语
            convlayer_3 = CnnLayer(in_channels=conv1_input_wordim, out_channels=conv1_input_wordim)
            conv3_result = convlayer_3.forward(inputs=conv2_result)

        # 执行每一层对应位置
        with tf.name_scope("vertical_attention"):
            all_word_rep = []
            all_word_rep.append(sentence_rep)
            all_word_rep.append(conv1_result)
            all_word_rep.append(conv2_result)
            all_word_rep.append(conv3_result)
            count = 0
            asp_pos_attentionlayer = DotAttentionLayer(edim=self.dim_w)
            for word_rep in getTimeStepsWord_tensor(inputs=all_word_rep,
                                                    batch_size=self.batch_size,
                                                    seq_length=self.sent_len,
                                                    word_dim=self.dim_w):
                # word_rep.shape = [batch_size, layers, word_dims]
                # 转化 aspects_rep.shape = [batch_size, dims, 1]
                aspects_rep = tf.reduce_mean(input_tensor=aspects_rep, axis=1, keep_dims=True)
                aspects_rep = tf.reshape(tensor=aspects_rep, shape=[-1, aspects_input_dim, 1])
                # attention 结果，表示该位置的rep
                vec, alpha = asp_pos_attentionlayer.forward(context=word_rep, aspect=aspects_rep)
                if count == 0:
                    step_vec = copy.copy(vec)
                    count += 1
                else:
                    step_vec = tf.concat(values=[step_vec, vec], axis=1)

        # step_vec.shape = [batch_size, steps, edims]
        with tf.name_scope("horizontal_attention"):
            asp_ctx_attentionlayer = DotAttentionLayer(edim=self.dim_w)
            ctx_vec, ctx_alpha = asp_ctx_attentionlayer.forward(context=step_rep, aspect=aspects_rep)

        # ctx_vec.shape = [batch_size, 1, conv1_input_wordim]
        # To calculate the softmax
        with tf.name_scope("softmax"):
            softmax_linear = Linearlayer(w_shape=[conv1_input_wordim, self.n_label],
                                         bias_shape=[1, self.n_label])
            # self.softmax_inputs.shape = [batch_size, 1, n_label] -> [batch_size, n_label]
            self.softmax_inputs = tf.reshape(tensor=softmax_linear.forward(inputs=ctx_vec),
                                             shape=[self.batch_size, self.n_label])
            pred_result = tf.nn.softmax(logits=self.softmax_inputs, name="predict_result")
            self.pred_result = tf.argmax(pred_result, 1)


    def build_loss(self):
        with tf.name_scope("loss"):
            y_shaped = tf.reshape(tensor=self.y, shape=self.softmax_inputs.shape)
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.softmax_inputs, labels=y_shaped)
            self.loss = tf.reduce_mean(loss)

    def build_optimizer(self):
        # clipping gradicent函数，对梯度进行clip梯度下降
        # tf.train_varibles
        train_var = tf.trainable_variables()
        # clip_by_global_norm是梯度缩放输入是所有trainable向量的梯度，和所有trainable向量，
        # 返回值 第一个为clip好的梯度，第二个为globalnorm
        # self.grad_clip 如何确定
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, train_var), self.grad_clip)
        # 构建一个优化器，此时学习率为learn_rate
        train_op = tf.train.AdamOptimizer(self.learn_rate)
        # 输入格式为grads, train_var, 来执行梯度的正式更新
        self.optimizer = train_op.apply_gradients(zip(grads, train_var))

    def __str__(self):
        return "CPAN"

    def get_modelresult(self, epoch,
                        n_train_batches,
                        n_test_batches,
                        train_set,
                        test_set,
                        embeddings):

        result_strings = []
        with tf.Session() as sess:
            for i in range(1, epoch + 1):

                # train-----
                print("In epoch %s/%s:" % (i, epoch))
                np.random.shuffle(train_set)
                train_y_pred, train_y_gold = [], []
                train_losses = []
                for j in range(n_train_batches):
                    start_time = time.time()
                    train_sent_id, train_target_id, train_y = batch_generate(train_set, n_train_batches, i)
                    feed_dict = {
                        self.inputs: train_sent_id,
                        self.aspects: train_target_id,
                        self.y: train_y,
                        self.embeddings: embeddings
                    }

                    loss, optimizer, pred, gold = sess.run(
                        fetches=[self.loss, self.optimizer, self.pred_result, self.y],
                        feed_dict=feed_dict)

                    end_time = time.time()
                    print('ith_batch/train_batch_num: {}/{}...'.format(i, n_train_batches),
                          'train_loss: {:.4f}...'.format(loss),
                          '{:.4f} sec/train_batch'.format((end_time - start_time)))
                    train_y_pred.append(pred)
                    train_y_gold.append(gold)
                    train_losses.append(loss)
                acc, f, _, _ = evaluate(pred=train_y_pred, gold=train_y_gold)
                print("\ttrain loss: %.4f, train acc: %.4f, train f1: %.4f" % (sum(train_losses), acc, f))

                # test------
                # test + train 速度太慢尝试储存模型
                test_y_pred, test_y_gold = [], []
                for k in range(n_test_batches):
                    test_sent_id, test_target_id, test_y = batch_generate(test_set, n_test_batches, i)
                    feed_dict = {
                        self.inputs: test_sent_id,
                        self.aspects: test_target_id,
                        self.y: test_y,
                        self.embeddings: embeddings
                    }
                    beg = time.time()
                    test_pred, test_gold = sess.run(
                        fetches=[self.pred_result, self.y],
                        feed_dict=feed_dict
                    )
                    end = time.time()
                    test_y_pred.extend(test_pred)
                    test_y_gold.extend(test_gold)
                # 检查 batch_size 还有 batch_num 以及 data的数目之间的关系
                #acc, f, _, _ = evaluate(pred=test_y_pred[:n_test], gold=test_y_gold[:n_test])
                acc, f, _, _ = evaluate(pred=test_y_pred, gold=test_y_gold)
                print("\tperformance of prediction: acc: %.4f, f1: %.4f" % (acc, f))
                result_strings.append("In Epoch %s: accuracy: %.2f, macro-f1: %.2f\n" % (i, acc * 100, f * 100))








