import tensorflow as tf
from tensorflow import keras


def rec_loss(S, F):
    result = (S-F)**2
    return tf.reduce_mean(tf.reduce_sum(result,[1,2]))

def T_reg(T):
    b_0 = tf.reduce_sum(tf.math.maximum(-1*T,0))
    b_1 = tf.reduce_sum(tf.math.maximum(T-1,0))
    return b_0+b_1

def W_reg(W):
    return tf.reduce_sum(tf.math.abs(W-1))

def stage1_loss(S,F,T,W):
    return 1.0 * rec_loss(S, F) + 1.0 * T_reg(T) + 1.0 * W_reg(W)

def stage2_loss(S,F,M):
    term1 = (1-F)*tf.abs(tf.minimum(S,1)-1)
    term2 = F*tf.abs(tf.maximum(S,0))
    term3 = tf.reduce_mean(M*F)
#     print(tf.matmul(M, tf.transpose(S, perm=[0, 2, 1])))
    result = tf.reduce_mean(term1 + term2 ) -  term3
    return result


