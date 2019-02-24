import tensorflow as tf

if __name__ == '__main__':

    # batch_size = 4
    # input = tf.random_normal(shape=[4, 3, 6], dtype=tf.float32)
    # cell = tf.nn.rnn_cell.BasicLSTMCell(10, forget_bias=1.0, state_is_tuple=True)
    # init_state = cell.zero_state(batch_size, dtype=tf.float32)
    # output, final_state = tf.nn.dynamic_rnn(cell, input, initial_state=init_state)

    #time_major如果是True，就表示RNN的steps用第一个维度表示，建议用这个，运行速度快一点。
    #如果是False，那么输入的第二个维度就是steps。
    #如果是True，output的维度是[steps, batch_size, depth]，反之就是[batch_size, max_time, depth]。就是和输入是一样的
    #final_state就是整个LSTM输出的最终的状态，包含c和h。c和h的维度都是[batch_size， n_hidden]

    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     #print(sess.run(output))
    #     #print(sess.run(final_state))
    #     print(sess.run(output))

    a = ['a', 'c', 't', 'e']
    b = a[::-1]
    print(b)

    tfinput = [[[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]],
               [[3, 2, 1],
                [6, 5, 4],
                [9, 8, 7]]]

    tfVar = tf.Variable(initial_value=tfinput, dtype=tf.float32)
    A = [1.0, 2.0, 3.0, 4.0, 5.0]
    As = tf.nn.softmax(A)
    tfV = [[2.0, 1.0, 3.0, 5.0, 4.0],
           [4.0, 5.0, 6.0, 7.0, 8.0],
           [8.0, 9.0, 0.0, 1.0, 2.0]]
    tfVA = tf.nn.softmax(tfV)
    tfpred = tf.argmax(tfVA, 1)
    asa = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        Asd = sess.run(As)
        print(sess.run(tfVA))
        tfpred = sess.run(tfpred)
        asa.extend(tfpred)
        print(asa)