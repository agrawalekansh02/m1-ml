import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import GridSearchCV
import keras
from keras.datasets import mnist
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten

# data
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
img_rows, img_cols = 28, 28
num_classes = 10
X_train = X_train[:40000]
Y_train = Y_train[:40000]
X_train = X_train.reshape(X_train.shape[0], img_rows*img_cols)
X_test = X_test.reshape(X_test.shape[0], img_rows*img_cols)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
Y_train = keras.utils.np_utils.to_categorical(Y_train, num_classes)
Y_test = keras.utils.np_utils.to_categorical(Y_test, num_classes)

# model
def DNN_200(activation_func):
    model = Sequential()
    model.add(Dense(200, input_shape=(img_rows*img_cols,), activation=activation_func))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])
    return model

batch_size = 64
epochs = 5
model_gridsearch = KerasClassifier(build_fn=DNN_200, 
                        epochs=epochs, batch_size=batch_size, verbose=1)

# scikit search module
activation_func = ['relu', 'sigmoid', 'softmax', 'tanh', 'selu']
param_grid = dict(activation_func=activation_func)
grid = GridSearchCV(estimator=model_gridsearch, param_grid=param_grid, n_jobs=1, cv=4)
grid_result = grid.fit(X_train, Y_train)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
