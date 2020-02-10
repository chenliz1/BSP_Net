import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,Sequential

class BasicBlock(tf.keras.layers.Layer):

    def __init__(self, filter_num, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(3, 3),
                                            strides=stride,
                                            padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(3, 3),
                                            strides=1,
                                            padding="same")
        self.bn2 = tf.keras.layers.BatchNormalization()
        if stride != 1:
            self.downsample = tf.keras.Sequential()
            self.downsample.add(tf.keras.layers.Conv2D(filters=filter_num,
                                                       kernel_size=(1, 1),
                                                       strides=stride))
            self.downsample.add(tf.keras.layers.BatchNormalization())
        else:
            self.downsample = lambda x: x

    def call(self, inputs, training=None, **kwargs):
        residual = self.downsample(inputs)

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)

        output = tf.nn.relu(tf.keras.layers.add([residual, x]))

        return output


def make_basic_block_layer(filter_num, blocks, stride=1):
    res_block = tf.keras.Sequential()
    res_block.add(BasicBlock(filter_num, stride=stride))

    for _ in range(1, blocks):
        res_block.add(BasicBlock(filter_num, stride=1))

    return res_block


class ResNet(keras.Model):
    def __init__(self,layer_dims,num_classes=256):
        super(ResNet, self).__init__()
        self.stem=Sequential([
            layers.Conv2D(64,7,strides=2,padding="same"),
            layers.BatchNormalization(),
            layers.Activation('relu')
            # ,
            # layers.MaxPool2D(pool_size=(3,3),strides=2,padding='same')
        ])
        # resblocks
        self.layer1 = make_basic_block_layer(64,layer_dims[0])
        self.layer2 = make_basic_block_layer(128, layer_dims[1],stride=2)
        self.layer3 = make_basic_block_layer(256, layer_dims[2], stride=2)
        self.layer4 = make_basic_block_layer(512, layer_dims[3], stride=2)

        # self.avgpool=layers.GlobalAveragePooling2D()
        self.fc=layers.Dense(units=num_classes)


    def call(self,input,training=None):
        x=self.stem(input)
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)

        # x=self.avgpool(x)
        x=self.fc(x)
        return x



def resnet18():
    return  ResNet([2,2,2,2])
# resnet18()