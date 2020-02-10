import tensorflow as tf
from tensorflow import keras


def rec_loss(S, F):
    result = (S-F)**2
    return tf.reduce_sum(tf.reduce_mean(result,[1,2]))

def T_reg(T):
    b_0 = tf.reduce_mean(tf.math.maximum(-1*T,0))
    b_1 = tf.reduce_mean(tf.math.maximum(T-1,0))
    return b_0+b_1

def W_reg(W):
    return tf.reduce_mean(tf.math.abs(W-1))

def stage1_loss(S,F,T,W):
    return 1.0 * rec_loss(S, F) + 1.0 * T_reg(T) + 1.0 * W_reg(W)


