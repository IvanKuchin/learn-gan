import tensorflow as tf
import numpy as np

# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def upsample():
    inputs = tf.keras.Input(shape=(2))
    dense = tf.keras.layers.Dense(2*2*3)(inputs)
    reshape = tf.keras.layers.Reshape((2, 2, 3))(dense)
    outputs = tf.keras.layers.UpSampling2D(interpolation="nearest")(reshape)
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    model.summary()

    img = np.asarray([[1,2]])
    yhat = model.predict(img)
    print(f"yhat shape: {yhat.shape}\nupsample: {yhat.squeeze()}")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    upsample()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
