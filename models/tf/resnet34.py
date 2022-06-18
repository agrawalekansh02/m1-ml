import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Layer, Conv2D, MaxPool2D, Dense, Activation, BatchNormalization, GlobalAveragePooling2D, Dropout
import random


class ResidualBlock(Layer):
    def __init__(self, block_type=None, n_filters=None):
        super(ResidualBlock, self).__init__()
        self.n_filters = n_filters
        if block_type == 'identity':
            self.strides = 1
        elif block_type == 'conv':
            self.strides = 2
            self.conv_shorcut = Conv2D(filters=self.n_filters, 
                               kernel_size=1, 
                               padding='same',
                               strides=self.strides,
                               kernel_initializer='he_normal')
            self.bn_shortcut = BatchNormalization()

        self.conv_1 = Conv2D(filters=self.n_filters, 
                               kernel_size=3, 
                               padding='same',
                               strides=self.strides,
                               kernel_initializer='he_normal',
                               name=f"conv_1{random.randint(0, 100)}")
        self.bn_1 = BatchNormalization()
        self.relu_1 = Activation('relu')

        self.conv_2 = Conv2D(filters=self.n_filters, 
                               kernel_size=3, 
                               padding='same', 
                               kernel_initializer='he_normal')
        self.bn_2 = BatchNormalization()
        self.relu_2 = Activation('relu')

    def call(self, x, training=False):
        shortcut = x
        if self.strides == 2:
            shortcut = self.conv_shorcut(x)
            shortcut = self.bn_shortcut(shortcut)
        y = self.conv_1(x)
        y = self.bn_1(y)
        y = self.relu_1(y)
        y = self.conv_2(y)
        y = self.bn_2(y)
        y = tf.add(shortcut, y)
        y = self.relu_2(y)
        return y

class ResNet34(Model):
    def __init__(self, latent, n_classes, activation):
        super(ResNet34, self).__init__()

        self.n_classes = n_classes
        self.latent = latent
        self.a = activation

        self.conv_1 = Conv2D(filters=64, 
                                kernel_size=7, 
                                padding='same', 
                                strides=2, 
                                kernel_initializer='he_normal')
        self.bn_1 = BatchNormalization(momentum=0.9)
        self.relu_1 = Activation('relu')
        self.maxpool = MaxPool2D(3, 2, padding='same')
        self.residual_blocks = []
        for n_filters, reps, downscale in zip([64, 128, 256, 512], 
                                              [3, 4, 6, 3], 
                                              [False, True, True, True]):
            for i in range(reps):
                if i == 0 and downscale:
                    self.residual_blocks.append(ResidualBlock(block_type='conv', 
                                                              n_filters=n_filters))
                else:
                    self.residual_blocks.append(ResidualBlock(block_type='identity', 
                                                              n_filters=n_filters))
        self.gap = GlobalAveragePooling2D()
        self.fc = Dense(self.latent, kernel_initializer='he_normal', activation="relu")
        self.o = Dense(units=self.n_classes, activation=self.a)

    def call(self, x, training=False):
        y = self.conv_1(x)
        y = self.bn_1(y)
        y = self.relu_1(y)
        y = self.maxpool(y)
        for layer in self.residual_blocks:
            y = layer(y)
        y = self.gap(y)
        y = self.fc(y)
        y = self.o(y)
        return y

    def summary(self, input_shape):
        x = Input(input_shape)
        model = Model(inputs=[x], outputs=self.call(x))
        return model.summary()