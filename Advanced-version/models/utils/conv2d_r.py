import tensorflow as tf
from keras.layers import Conv2D, Lambda
from .sn import ConvSN2D

def Conv2D_r(channels, filter_size, strides, features):
    padding = [[0, 0], [filter_size // 2, filter_size // 2],
               [filter_size // 2, filter_size // 2], [0, 0]]
    
    out = Lambda(lambda net: tf.pad(net, padding, 'REFLECT'))(features)
    out = ConvSN2D(channels, filter_size, strides=strides, padding='valid')(out)
    return out