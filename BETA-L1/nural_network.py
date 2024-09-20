import tensorflow as tf
from keras import layers
import Gloabl_variable as gb

feature1 = 128
feature2 = 256
def get_actor():
    # Initialize weights between -3e-3 and 3-e3
    last_init = tf.random_uniform_initializer(minval=-0.03, maxval= 0.03)

    inputs = layers.Input(shape=(gb.num_states, gb.state_len))

    slice1 = layers.Lambda(lambda x: x[:, 0:1, -1])(inputs)
    slice2 = layers.Lambda(lambda x: x[:, 1:2, -1])(inputs)
    slice3 = layers.Lambda(lambda x: x[:, 2:3, :])(inputs)
    slice4 = layers.Lambda(lambda x: x[:, 3:4, :])(inputs)
    slice5 = layers.Lambda(lambda x: x[:, 4:5, -1])(inputs)

    conv3 = layers.Conv1D(filters=feature1, kernel_size=3, padding='same')(slice3)
    conv4 = layers.Conv1D(filters=feature1, kernel_size=3, padding='same')(slice4)

    flatten_conv3 = layers.Flatten()(conv3)
    flatten_conv4 = layers.Flatten()(conv4)
    dens1 = layers.Dense(feature1, activation="relu")(slice1)
    dens2 = layers.Dense(feature1, activation="relu")(slice2)
    dens5 = layers.Dense(feature1, activation="relu")(slice5)

    added = layers.Concatenate()([dens1, dens2, flatten_conv3, flatten_conv4, dens5])
    added = layers.Flatten()(added)
    out = layers.Dense(feature2, activation="relu")(added)
    out = layers.Dense(1, activation="tanh", kernel_initializer=last_init)(out)

    model = tf.keras.Model(inputs, out)
    return model


def get_critic():
    # State as input

    inputs = layers.Input(shape=(gb.num_states, gb.state_len))
    slice1 = layers.Lambda(lambda x: x[:, 0:1, -1])(inputs)
    slice2 = layers.Lambda(lambda x: x[:, 1:2, -1])(inputs)
    slice3 = layers.Lambda(lambda x: x[:, 2:3, :])(inputs)
    slice4 = layers.Lambda(lambda x: x[:, 3:4, :])(inputs)
    slice5 = layers.Lambda(lambda x: x[:, 4:5, -1])(inputs)

    conv3 = layers.Conv1D(filters=feature1, kernel_size=3, padding='same')(slice3)
    conv4 = layers.Conv1D(filters=feature1, kernel_size=3, padding='same')(slice4)
    flatten_conv3 = layers.Flatten()(conv3)
    flatten_conv4 = layers.Flatten()(conv4)
    dens1 = layers.Dense(feature1, activation="relu")(slice1)
    dens2 = layers.Dense(feature1, activation="relu")(slice2)
    dens5 = layers.Dense(feature1, activation="relu")(slice5)

    action_input = layers.Input(shape=gb.num_actions)
    action_out = layers.Dense(feature1, activation="relu")(action_input)
    added = layers.Concatenate()([dens1, dens2, flatten_conv3, flatten_conv4, dens5, action_out])
    out = layers.Dense(feature2, activation="relu")(added)
    outputs = layers.Dense(1)(out)

    model = tf.keras.Model([inputs, action_input], outputs)

    return model
