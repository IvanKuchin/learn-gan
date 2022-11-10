import tensorflow as tf
# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def make_net():
    inputs = tf.keras.Input(shape=(2,), name="Input")
    hidden1 = tf.keras.layers.Dense(10, activation="relu")(inputs)
    hidden2 = tf.keras.layers.Dense(20, activation="relu")(hidden1)
    hidden3 = tf.keras.layers.Dense(10, activation="relu")(hidden2)
    output = tf.keras.layers.Dense(1, activation="sigmoid")(hidden3)

    model = tf.keras.Model(inputs=inputs, outputs=[output])
    model.summary()

    tf.keras.utils.plot_model(model, show_shapes=True)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    make_net()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
