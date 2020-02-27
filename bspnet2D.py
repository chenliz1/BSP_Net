from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
from encoder2D import encoder2D
from decoder2D import decoder2D
import numpy as np

class BspNet2D(keras.Model):
    def __init__(self, batchsize):
        super(BspNet2D, self).__init__()
        self.encoder = encoder2D()

        A = np.arange(64).reshape(1, 64) + 0.5
        A = np.repeat(A, 64, axis=0)
        B = A.T.reshape(-1, 1) + 0.5
        A = np.concatenate((A.reshape(-1, 1), B, np.ones(B.shape), np.ones(B.shape)), 1)
        points = tf.cast(tf.constant(A),dtype=tf.float32)


        self.decoder = decoder2D(points, batchsize=batchsize)


    def call(self,input,training=None):

        x=self.encoder(input)
        x=self.decoder(x)

        return x