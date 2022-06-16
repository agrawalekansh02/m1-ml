from numpy import block
import tensorflow as tf

def ResNet34(input_shape, num_classes, filter_size=64, activation="relu", kernel_size=(3,3)):
    input_l = tf.keras.layers.Input(shape=input_shape)

    # layer 1
    conv1 = tf.keras.layers.Conv2D(filters=64, 
                                kernel_size=7,
                                stride=2,
                                padding="same",
                                name="conv1")(input_l)
    batch_norm = tf.keras.layers.BatchNormalization(name="batch_norm1")(conv1)
    activation_1 = tf.keras.layers.Activation(activation, name="activation1")(batch_norm)
    x = tf.keras.layers.MaxPool2D(pool_size=3, 
                                strides=2,
                                padding="same",
                                name="pool1")(activation_1)

    # resnet blocks
    block_layers = [3, 4, 6, 3]
    counter = 0
    for i in range(4):
        if i == 0:
            for j in range(block_layers[i]):
                x = res_block_iden(counter, x, filter_size)
                counter += 1
        else:
            filter_size *= 2
            x = res_block_cnn(counter, x, filter_size)
            counter += 1
            for j in range(block_layers[i]-1):
                x = res_block_iden(counter, x, filter_size)
                counter += 1

    # dense connections
    avg_pool = tf.keras.layers.AveragePooling2D(pool_size=2,
                                        padding='same',
                                        name="avg_pool")(x)
    flatten = tf.keras.layers.Flatten(name="flatten")(avg_pool)
    dense = tf.keras.layers.Dense(2048, 
                                activation=activation, 
                                name="dense")(flatten)
    output_l = tf.keras.layers.Dense(num_classes, 
                                    activation="softmax", 
                                    name="output")(dense)
    model = tf.keras.Model(inputs=input_l, outputs=output_l, name="resnet34")
    return model



def res_block_iden(i, x, filters, activation='relu', kernel_size=(3,3)):
    x_copy = x
    
    # layer 1
    conv_1 = tf.keras.layers.Conv2D(filters, 
                                    kernel_size=kernel_size, 
                                    padding="same",
                                    name=f"resblock{i}_l1")(x)
    batch_norm_1 = tf.keras.layers.BatchNormalization(axis=3)(conv_1)
    activation_1 = tf.keras.layers.Activation(activation, 
                                    name=f"activation{i}_l1")(batch_norm_1)
    
    # layer 2
    conv_2 = tf.keras.layers.Conv2D(filters,
        kernel_size=kernel_size,
        padding="same",
        name=f"resblock{i}_l2")(activation_1)
    batch_norm_1 = tf.keras.layers.BatchNormalization(axis=3)(conv_2)

    # residuals
    residual = tf.keras.layers.Add()([x_copy, conv_2])
    output = tf.keras.layers.Activation(activation)(residual)
    return output

def res_block_cnn(i, x, filters, activation='relu', kernel_size=(3,3)):
    x_copy = x
    
    # layer 1
    conv_1 = tf.keras.layers.Conv2D(filters, 
                                    kernel_size=kernel_size, 
                                    strides=2,
                                    padding="same", 
                                    name=f"resblock{i}_l1")(x)
    batch_norm_1 = tf.keras.layers.BatchNormalization(axis=3)(conv_1)
    activation_1 = tf.keras.layers.Activation(activation,
                                    name=f"activation{i}_l1")(batch_norm_1)

    # layer 2
    conv_2 = tf.keras.layers.Conv2D(filters,
                                    kernel_size=kernel_size,
                                    padding="same",
                                    name=f"resblock{i}_l2")(activation_1)
    batch_norm_2 = tf.keras.layers.BatchNormalization(axis=3)(conv_2)

    # residuals
    conv_3 = tf.keras.layers.Conv2D(filters,
                                    kernel_size=1,
                                    strides=2,
                                    name=f"resblock{i}_l3")(x_copy)
    residual = tf.keras.layers.Add()([batch_norm_2, conv_3])
    output = tf.keras.layers.Activation(activation)(residual)
    return output