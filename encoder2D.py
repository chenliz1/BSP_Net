import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,Sequential


class encoder2D(keras.Model):
    def __init__(self,num_classes=256):
        super(encoder2D, self).__init__()
        self.stem=Sequential([
            layers.Conv2D(16,3,strides=2,padding="same", input_shape=(64, 64, 1)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(32,3,strides=2,padding="same"),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(64, 3, strides=2, padding="same"),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(128, 3, strides=2, padding="same"),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Flatten(),
            layers.Dense(units=num_classes),
            layers.Activation('relu')
            # layers.Conv2D(256, 3, strides=2, padding="same"),
        ])



    def call(self,input,training=None):

        x=self.stem(input)

        return x
