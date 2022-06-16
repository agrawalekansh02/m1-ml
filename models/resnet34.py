import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Layer, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Activation, BatchNormalization

class ResidualBlock(Layer):
    def __init__(self, filter_size, block_type, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)

        self.f = filter_size
        if block_type == "iden":
            self.s = 1
        elif block_type == "conv":
            self.s = 2

    
    def build(self):
        if self.s == 1:
            self.conv_shorcut = Conv2D(filters=self.f, 
                                kernel_size=1, 
                                padding='same',
                                strides=self.s)
            self.bn_shortcut = BatchNormalization()

        self.conv_1 = Conv2D(filters=self.f, 
                               kernel_size=3, 
                               padding='same',
                               strides=self.s)
        self.bn_1 = BatchNormalization()
        self.relu_1 = Activation('relu')

        self.conv_2 = Conv2D(filters=self.f, 
                               kernel_size=3, 
                               padding='same')
        self.bn_2 = BatchNormalization()
        self.relu_2 = Activation('relu')

    
    def call(self, x):
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
    def __init__(self, final_activation, latent_dim, num_classes, **kwargs):
        super(ResNet34, self).__init__(**kwargs)

        self.a = final_activation
        self.l = latent_dim
        self.nc = num_classes

    
    def build(self):
        self.conv = Sequential()
        self.conv.add(Conv2D(filters=64, 
                        kernel_size=7, 
                        stride=2, 
                        padding="same"))
        self.conv.add(BatchNormalization())
        self.conv.add(Activation("relu"))
        self.conv.add(MaxPooling2D(pool_size=3, 
                        strides=2, 
                        padding="same"))

        for n, layers, downscale in zip([64, 128, 256, 512], 
                                        [3, 4, 6, 3], 
                                        [False, True, True, True]):
            for i in range(layers):
                if i == 0 and downscale:
                    self.conv.add(ResidualBlock(n, "iden"))
                else:
                    self.conv.add(ResidualBlock(n, "conv"))
        
        self.conv.add(AveragePooling2D())
        self.conv.add(Flatten())
        self.conv.add(Dense(self.l, activation="relu"))
        self.conv.add(Dense(self.nc, activation=self.a))

    def call(self, x):
        return self.conv(x)

