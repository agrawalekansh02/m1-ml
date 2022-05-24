import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.datasets import cifar10
from resnet_model import *

# mnist data
# (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
# img_rows, img_cols = 28, 28
# num_classes = 10
# X_train = X_train
# Y_train = Y_train
# X_train = X_train.reshape(X_train.shape[0], img_rows*img_cols)
# X_test = X_test.reshape(X_test.shape[0], img_rows*img_cols)
# X_train = X_train.astype('float64')
# X_test = X_test.astype('float64')
# X_train /= 255
# X_test /= 255
# Y_train = keras.utils.np_utils.to_categorical(Y_train, num_classes)
# Y_test = keras.utils.np_utils.to_categorical(Y_test, num_classes)

# load cifar data
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
print(X_train.shape)
# img_rows, img_cols = 28, 28
# num_classes = 10
# X_train = X_train
# Y_train = Y_train
# X_train = X_train.reshape(X_train.shape[0], img_rows*img_cols)
# X_test = X_test.reshape(X_test.shape[0], img_rows*img_cols)
# X_train = X_train.astype('float64')
# X_test = X_test.astype('float64')
# X_train /= 255
# X_test /= 255
# Y_train = keras.utils.np_utils.to_categorical(Y_train, num_classes)
# Y_test = keras.utils.np_utils.to_categorical(Y_test, num_classes)

# testing resnet34 with relu
model = ResNet34(input_shape=(180, 180, 3), num_classes=num_classes)
print(model.summary())
# model.fit(X_train, Y_train, epochs=20, batch_size=128, validation_split=0.2)