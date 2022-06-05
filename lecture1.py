import tensorflow as tf

class MyDenseLayer(tf.keras.layers.layer):
    def __init__(self, input_dim, output_dim) -> None:
        super(MyDenseLayer, self).__init__()

        self.weights = self.add_weight([input_dim, output_dim])
        self.bias = self.add_weight([1, output_dim])

    def call(self, inputs):
        # Forward propagate the inputs
        z = tf.matmul(inputs, self.weights) + self.bias

        output = tf.math.sigmoid(z)

        return output