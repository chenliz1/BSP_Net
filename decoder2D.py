import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,Sequential
import numpy as np


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
    
class L2LayerStage2(tf.keras.layers.Layer):
    def __init__(self, T, p_value, c_value):
        super(L2LayerStage2, self).__init__()
        self.c_value = c_value
        self.p_value = p_value
        self.weight = tf.Variable(initial_value=T, dtype=tf.float32, trainable=True)

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
        return tf.keras.backend.clip(tf.matmul(x, self.weight), 0, 1)


class decoder2D(keras.Model):
    def __init__(self, points, stage=1, p_value=256, c_value=64, batchsize=32):
        super(decoder2D, self).__init__()
        self.layer0=Sequential([
            layers.Dense(units=512, input_shape=(p_value,)),
            layers.Dense(units=1024),
            layers.Dense(units=2048),
            layers.Dense(units=4 * p_value),
            layers.Reshape((p_value, 4), input_shape=(4 * p_value,))
        ])
        self.stage = stage
        self.c_value = c_value
        self.p_value = p_value
        self.points = points
        self.L2 = L2Layer(p_value, c_value)
        self.L3 = L3Layer(c_value)


    def call(self,input,training=None):
        batch_size = input.shape[0]
        x=self.layer0(input)#P
        x = tf.transpose(x, perm=[0, 2, 1])#P.trans

        x = tf.tile(self.points[None], [batch_size, 1, 1]) @ x #D=xP.trans
        
        
        x2 = self.L2(x)
        if self.stage==1:

            x3 = self.L3(x2)
            return x3, x2
        else:
            x3 = tf.reduce_min(x2, axis=2, keepdims=True)
            M = tf.math.count_nonzero(x2, axis=2,keepdims=True)
            M = tf.cast(M, tf.float32)

            
            M = self.c_value - M - 1.0
            
            M = tf.keras.backend.clip(M, 0.0, 1.0)

            
       
            return x3, M
    
    def switchStage(self, stage):
        self.stage = stage
        if stage==2:
            T = self.L2.weight.numpy()
            t = np.zeros(T.shape)
            t[T > 0.01] = 1.0
            t[T<=0.01] = 0.0
            self.L2.weight = tf.Variable(initial_value=t, dtype=tf.float32, trainable=True)
            w = self.L3.weight.numpy()
            self.L3.weight = tf.Variable(initial_value=w, dtype=tf.float32, trainable=False)
        else:
            w = self.L3.weight.numpy()
            self.L3.weight = tf.Variable(initial_value=w, dtype=tf.float32, trainable=True)

class decoder2DStage2(keras.Model):
    def __init__(self, T, p_value=256, c_value=64, batchsize=32):
        super(decoder2DStage2, self).__init__()
        t = np.zeros(T.shape)
        t[T > 0.01] = 1.0
        t[T<=0.01] = 0.0
        self.c_value = c_value

        self.L2 = L2LayerStage2(t, p_value, c_value)

    def call(self,input,training=None):
        
        x2 = self.L2(input)
        M = (self.c_value - tf.math.count_nonzero(x2, axis=2,keepdims=True)) - 1
        M = tf.keras.backend.clip(M, 0, 1)
        M = tf.cast(M, tf.float32)
        x3 = tf.math.reduce_min(x2, axis=2, keepdims=True)

        return x3, M