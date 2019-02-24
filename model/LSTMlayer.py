import tensorflow as tf

class LSTMlayer():

    def __init__(
        self,
        hidden_size,
        output_keep_prob = 0.8,
        input_keep_prob = 1.0,
        forget_bias = 1.0,
        cell = "lstm"
    ):
        self.hidden_size = hidden_size,
        self.output_keep_prob = output_keep_prob
        self.input_keep_prob = input_keep_prob
        self.forget_bias = forget_bias
        print(hidden_size)
        print(self.hidden_size)
        self.lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(
            num_units=hidden_size,
            forget_bias=1.0,
            state_is_tuple=True
        )
        # self.lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
        #     cell=self.lstm_cell,
        #     input_keep_prob=input_keep_prob,
        #     output_keep_prob=output_keep_prob
        # )


    # inputs.shape = [batch_size, steps, edims]
    # results = [batch_size, sequence_length, edims]
    # state = [batch_size, hidden_size]
    def forward(self, inputs):
        # inputs = tf.random_normal(shape=[4, 3, 6], dtype=tf.float32)
        batch_size = tf.shape(inputs)[0]
        print(batch_size)
        steps = tf.shape(inputs)[1]
        print(steps)
        init_state = self.lstm_cell.zero_state(batch_size=batch_size, dtype=tf.float32)
        results, state = tf.nn.dynamic_rnn(cell=self.lstm_cell, inputs=inputs, initial_state=init_state)
        return results, state