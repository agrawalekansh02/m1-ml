from sklearn.semi_supervised import SelfTrainingClassifier
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


def ModifiedVGG(Model):
    def __init__(self, layer_activation, final_activation, dropout_rate, latent_dim, num_classes):
        super(ModifiedVGG, self).__init__()

        self.al = layer_activation
        self.af = final_activation
        self.d = dropout_rate
        self.l = latent_dim
        self.nc = num_classes


    def build(self):
        self.conv = Sequential()
        for filters, num_layers in zip([64, 128, 256], [2, 2, 3]):
            for _ in range(num_layers):
                self.conv.add(Conv2D(filters, (3, 3), padding='same', activation=self.al))
            self.conv.add(MaxPooling2D((2, 2)))
            self.conv.add(Dropout(self.d))

        self.flatten = Flatten()
        self.fc1 = Dense(self.l, activation=self.al)
        self.fc2 = Dense(self.l, activation=self.al)
        self.o = Dense(self.nc, activation=self.af)


    def call(self, input):
        x = self.conv(input)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.o(x)
        return x



