import io

import tensorflow as tf
import numpy as np
import datetime
from matplotlib import pyplot

from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy import vstack
from numpy.random import randn
from numpy.random import randint
from keras.datasets.mnist import load_data
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout
from matplotlib import pyplot


def define_discriminator():
    inputs = tf.keras.Input(shape=(28, 28, 1), name="inputs")
    conv1 = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(2, 2), padding="same", name="conv1")(inputs)
    bn1 = tf.keras.layers.BatchNormalization()(conv1)
    act1 = tf.keras.layers.LeakyReLU(alpha=0.2)(conv1)
    drop1 = tf.keras.layers.Dropout(0.4)(act1)
    conv2 = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="leaky_relu",
                                   name="conv2")(drop1)
    bn2 = tf.keras.layers.BatchNormalization()(conv2)
    act2 = tf.keras.layers.LeakyReLU(alpha=0.2)(conv2)
    drop2 = tf.keras.layers.Dropout(0.4)(act2)
    flat = tf.keras.layers.Flatten()(drop2)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(flat)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="disc")

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    return model


def define_generator(input_dims):
    inputs = tf.keras.Input(shape=input_dims)
    hidden1 = tf.keras.layers.Dense(7 * 7 * 128, activation="leaky_relu", name="dense1")(inputs)
    hidden2 = tf.keras.layers.Reshape((7, 7, 128))(hidden1)
    conv1 = tf.keras.layers.Conv2DTranspose(128, kernel_size=(4, 4), strides=(2, 2), padding="same", name="conv1tr")(
        hidden2)
    bn1 = tf.keras.layers.BatchNormalization()(conv1)
    act1 = tf.keras.layers.LeakyReLU(alpha=0.2)(conv1)
    conv2 = tf.keras.layers.Conv2DTranspose(64, kernel_size=(4, 4), strides=(2, 2), padding="same", name="conv2tr")(
        act1)
    bn2 = tf.keras.layers.BatchNormalization()(conv2)
    act2 = tf.keras.layers.LeakyReLU(alpha=0.2)(conv2)
    outputs = tf.keras.layers.Conv2D(1, kernel_size=(7, 7), padding="same", activation="sigmoid", name="conv3")(act2)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="generator")

    return model


def define_gan(disc, gen, input_dims):
    disc.trainable = False

    inputs = tf.keras.Input(shape=input_dims)
    g_out = gen(inputs)
    d_out = disc(g_out)

    model = tf.keras.Model(inputs=inputs, outputs=d_out, name="gan")

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    return model


def generate_real_samples(dataset, batch):
    ix = np.random.randint(0, dataset.shape[0], batch)
    X = dataset[ix]
    X = np.expand_dims(X, axis=-1)

    y = np.ones((batch, 1))
    return (X, y)


def generate_latent_points(input_dims, batch):
    r = np.random.random(batch * input_dims) * 2 - 1
    return np.reshape(r, (batch, input_dims))


def generate_fake_samples(gen, input_dims, batch):
    latent_points = generate_latent_points(input_dims, batch)
    X = gen.predict(latent_points, verbose=0)

    y = np.zeros((batch, 1))
    return (X, y)


def save_sample_pic(ds, name):
    ix = np.random.randint(0, ds.shape[0], 16)
    picset = ds[ix]

    for i in range(picset.shape[0]):
        pyplot.subplot(4, 4, i + 1)
        pyplot.axis("off")
        pyplot.imshow(picset[i], cmap="Greys")
        # print(np.average(picset[i]))

    pyplot.savefig(f"progress/{name}.png")

    # save image to memory
    buf = io.BytesIO()
    pyplot.savefig(buf, format="png")
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=1)
    image = np.expand_dims(image, 0)

    pyplot.close()
    return image

def train_disc(disc, gen, dataset, input_dims, batch):
    (realX, realY) = generate_real_samples(dataset, batch // 2)
    (fakeX, fakeY) = generate_fake_samples(gen, input_dims, batch // 2)

    X, Y = np.vstack((realX, fakeX)), np.vstack((realY, fakeY))

    loss, _ = disc.train_on_batch(X, Y)

    return loss


def summarize_performance(disc, gen, dataset, epoch, input_dims, batch):
    # latent_points = generate_latent_points(input_dims, 16)
    # ds = gen.predict(latent_points, verbose=0)
    (ds, _) = generate_fake_samples(gen, input_dims, 16)
    ds = np.squeeze(ds)
    image = save_sample_pic(ds, f"{epoch:05d}")

    (realX, realY) = generate_real_samples(dataset, batch // 2)
    (fakeX, fakeY) = generate_fake_samples(gen, input_dims, batch // 2)

    loss_real, acc_real = disc.evaluate(realX, realY, verbose=0)
    loss_fake, acc_fake = disc.evaluate(fakeX, fakeY, verbose=0)

    return acc_real, acc_fake, image


def train_gan(gan, input_dims, batch):
    fakeX = generate_latent_points(input_dims, batch)
    realY = np.ones((batch, 1))
    loss, _ = gan.train_on_batch(fakeX, realY)

    return loss


def train(disc, gen, gan, dataset, n_epochs, input_dims, batch, eval_frequency):
    disc_losses = []
    gan_losses = []

    ((trainX, trainy), (testX, testy)) = dataset
    n_iter = trainX.shape[0] // batch

    curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = f"logs/{curr_time}/train"
    test_log_dir = f"logs/{curr_time}/test"
    train_writer = tf.summary.create_file_writer(train_log_dir)
    test_writer = tf.summary.create_file_writer(test_log_dir)

    for i in range(n_epochs):
        for j in range(n_iter):
        # for j in range(3):
            disc_loss = train_disc(disc, gen, trainX, input_dims, batch)
            gan_loss = train_gan(gan, input_dims, batch)

            disc_losses.append(disc_loss)
            gan_losses.append(gan_loss)

            print(f"{i}, {j}/{n_iter}: {disc_loss=:.3f}, {gan_loss=:.3f}")

            with train_writer.as_default():
                tf.summary.scalar("disc_loss", disc_loss, step=i * n_iter + j)
                tf.summary.scalar("gen_loss", gan_loss, step=i * n_iter + j)

            save_trainable_vars(model=disc, writer=train_writer, step=i * n_iter + j)

        if i % eval_frequency == 0:
            (acc_real, acc_fake, image) = summarize_performance(disc, gen, testX, i, input_dims, batch)
            print(f"{acc_real=}\t{acc_fake=}")

            with test_writer.as_default():
                tf.summary.scalar("accuracy_real", acc_real, step=i)
                tf.summary.scalar("accuracy_fake", acc_fake, step=i)
                tf.summary.image("generator", image, step=i)

    pyplot.plot(disc_losses)
    pyplot.plot(gan_losses)
    pyplot.savefig("progress/losses.png")
    pyplot.close()


def save_trainable_vars(model, writer, step):
    layers = model.weights

    with writer.as_default():
        for layer in layers:
            if "kernel" in layer.name:
                # print("model: %10s | layer: %20s | shape: %s" % (model.name, layer.name, layer.shape))
                weights = layer.numpy().reshape(-1)
                tf.summary.histogram(name=f"{model.name}_{layer.name}", data=weights, step=step, buckets=100, description=f"{model.name} {layer.name} kernels")


if __name__ == '__main__':
    latent_dims = 100
    epochs = 5
    batch_size = 256
    eval_freq = 1

    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    save_sample_pic(X_train, "init")

    d = define_discriminator()
    g = define_generator(latent_dims)
    ga = define_gan(d, g, latent_dims)
    # ga = define_gan(d, g)

    train(d, g, ga, ((X_train, y_train), (X_test, y_test)), epochs, latent_dims, batch_size, eval_freq)
