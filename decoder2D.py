import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,Sequential


class L2Layer(tf.keras.layers.Layer):
    def __init__(self, p_value, c_value):
        super(L2Layer, self).__init__()
        self.c_value = c_value
        self.p_value = p_value
        initializer = tf.random_normal_initializer(mean=0.0, stddev=0.02)
        self.weight = tf.Variable(initial_value=initializer(
            shape=(p_value, c_value), dtype=tf.float32), trainable=True)

    def call(self, input):
        x = tf.keras.activations.relu(input)
        return tf.matmul(x, self.weight)


class L3Layer(tf.keras.layers.Layer):
    def __init__(self, c_value):
        super(L3Layer, self).__init__()
        self.c_value = c_value

        initializer = tf.random_normal_initializer(mean=0.0, stddev=0.02)
        self.weight = tf.Variable(initial_value=initializer(
            shape=(c_value, 1), dtype=tf.float32), trainable=True)

    def call(self, input):
        x = 1 - input
        x = tf.keras.backend.clip(x, 0, 1)
        return tf.matmul(x, self.weight)


class decoder2D(keras.Model):
    def __init__(self, points, p_value=256, c_value=64, batchsize=32):
        super(decoder2D, self).__init__()
        self.layer0=Sequential([
            layers.Dense(units=512, input_shape=(p_value,)),
            layers.Dense(units=1024),
            layers.Dense(units=2048),
            layers.Dense(units=3 * p_value),
            layers.Reshape((p_value, 3), input_shape=(3 * p_value,))
        ])

        self.points = points
        self.L2 = L2Layer(p_value, c_value)
        self.L3 = L3Layer(c_value)


    def call(self,input,training=None):
        batch_size = input.shape[0]
        x=self.layer0(input)
        x = tf.transpose(x, perm=[0, 2, 1])
        x = tf.tile(self.points[None], [batch_size, 1, 1]) @ x
        x = self.L2(x)
        x = self.L3(x)

        return x

# decoder2D()