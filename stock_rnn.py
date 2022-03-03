import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import keras
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.models import load_model

# loading
print('loading data')
df = pd.read_csv('AAPL.csv')
labels = pd.DataFrame(df['Adj Close'])
features_titles = ['Open', 'High', 'Low', 'Volume']

# scaling
scaler = MinMaxScaler()
feature_transform = scaler.fit_transform(df[features_titles])
features = pd.DataFrame(columns=features_titles, data=feature_transform, index=df.index)

# splitting to training and testing set
print('splitting data')
timesplit = TimeSeriesSplit(n_splits=10)
for train_index, test_index in timesplit.split(feature_transform):
        x_train, x_test = feature_transform[:len(train_index)], feature_transform[len(train_index): (len(train_index)+len(test_index))]
        y_train, y_test = labels[:len(train_index)].values.ravel(), labels[len(train_index): (len(train_index)+len(test_index))].values.ravel()

# preprocessing for rnn
print('preprocessing data')
trainX = np.array(x_train)
testX = np.array(x_test)
x_train = trainX.reshape(x_train.shape[0], 1, x_train.shape[1])
x_test = testX.reshape(x_test.shape[0], 1, x_test.shape[1])

# model building
print('building model')
def lstm():
    model = Sequential()
    model.add(LSTM(32, input_shape=(1, trainX.shape[1]), 
        activation='relu', 
        return_sequences=False))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam') 
    return model

rnn = lstm()

# training
print('training model')
history = rnn.fit(x_train, y_train, epochs=100, batch_size=8, verbose=1, shuffle=False)

# prediction
y_pred = rnn.predict(x_test)

# testing
plt.figure()
plt.plot(y_test, label='True Value')
plt.plot(y_pred, label='LSTM Value')
plt.title("Prediction by LSTM")
plt.xlabel('Time Scale')
plt.ylabel('Scaled USD')
plt.legend()
plt.show()