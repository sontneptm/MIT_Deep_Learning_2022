import tensorflow as tf
from tensorflow.keras.layers import Layer

class MyRNNCell(Layer):
    def __init__(self, rnn_units, input_dim, output_dim):
        super(MyRNNCell, self).__init__()

        self.W_xh = self.add_weight([rnn_units, input_dim])
        self.W_hh = self.add_weight([rnn_units, rnn_units])
        self.W_xy = self.add_weight([output_dim, rnn_units])

        self.h =tf.zeros([rnn_units, 1])

    def call(self, x):
        self.h = tf.math.tanh(self.w_hh * self.h + self.W_xh* x)

        output = self.W_hy * self.h

        return output, self.h