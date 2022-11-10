import matplotlib.pyplot
import tensorflow as tf
import numpy as np
# from matplotlib import pyplot


def approximate_func(x):
    return np.sin(x)


def generate_real_samples(n):
    x = np.random.random((n, 1)) * 6 - 3
    y = approximate_func(x)
    inputs = np.hstack((x, y))
    outputs = np.ones((n, 1))
    return inputs, outputs


def generate_fake_samples(n, generator, latent_dims):
    g_input = np.random.random((n, latent_dims))
    X = generator.predict(g_input, verbose=0)
    y = np.zeros((n, 1))
    return X, y

def generate_latent_samples(latent_dims, n):
    return np.random.random((n, latent_dims))


def define_discriminator(input_dims):
    inputs = tf.keras.Input(shape=input_dims)
    hidden1 = tf.keras.layers.Dense(20, activation="relu", kernel_initializer="he_uniform")(inputs)
    hidden2 = tf.keras.layers.Dense(10, activation="relu", kernel_initializer="he_uniform")(hidden1)
    hidden3 = tf.keras.layers.Dense(10, activation="relu", kernel_initializer="he_uniform")(hidden2)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", kernel_initializer="he_uniform")(hidden3)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="discriminator")
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"], loss_weights=[0.5])

    return model


def define_generator(latent_dims, input_dims):
    inputs = tf.keras.Input(shape=latent_dims)
    hidden1 = tf.keras.layers.Dense(25, activation="leaky_relu", kernel_initializer="he_uniform")(inputs)
    hidden2 = tf.keras.layers.Dense(20, activation="leaky_relu", kernel_initializer="he_uniform")(hidden1)
    hidden3 = tf.keras.layers.Dense(20, activation="leaky_relu", kernel_initializer="he_uniform")(hidden2)
    outputs = tf.keras.layers.Dense(domain_dims, activation="linear", kernel_initializer="he_uniform")(hidden3)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="generator")

    return model


def define_gan(discriminator, generator, latent_dims):
    discriminator.trainable=False
    inputs = tf.keras.Input(shape = latent_dims)
    hidden1 = generator(inputs)
    outputs = discriminator(hidden1)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer = "adam", loss="binary_crossentropy")

    return model


def summarize_performance(discriminator, generator, gan, epoch, latent_dims, n):
    (X_real, y_real) = generate_real_samples(n)
    (X_fake, y_fake) = generate_fake_samples(n, generator, latent_dims)

    loss, acc_real = discriminator.evaluate(X_real, y_real, verbose=0)
    loss, acc_fake = discriminator.evaluate(X_fake, y_fake, verbose=0)

    matplotlib.pyplot.scatter(X_real[:, 0], X_real[:,  1], c="green")
    matplotlib.pyplot.scatter(X_fake[:, 0], X_fake[:,  1], c="red")
    matplotlib.pyplot.savefig(f"progress\{epoch:05d}.png")
    matplotlib.pyplot.close()

    print(f"{epoch}: acc_real: {acc_real}, acc_fake: {acc_fake}")
    return (epoch, acc_real, acc_fake)

def train_discriminator(discriminator, generator, latent_dims, n):

    (X_real, y_real) = generate_real_samples(n)
    (X_fake, y_fake) = generate_fake_samples(n, generator, latent_dims)

    discriminator.train_on_batch(X_real, y_real)
    discriminator.train_on_batch(X_fake, y_fake)


def train_gan(g, latent_dims, n):
    X_latent = generate_latent_samples(latent_dims, n)
    y_real = np.ones((n, 1))

    g.train_on_batch(X_latent, y_real)


def train(discriminator, generator, gan, latent_dims, n_epochs, n, n_eval):
    epochs = []
    accs_real = []
    accs_fake = []

    for i in range(n_epochs):
        train_discriminator(discriminator, generator, latent_dims, n // 2)
        train_gan(gan, latent_dims, n)

        if i % n_eval == 0:
            (epoch, acc_real, acc_fake) = summarize_performance(discriminator, generator, gan, i, latent_dims, n)
            accs_real.append(acc_real)
            accs_fake.append(acc_fake)
            epochs.append(epoch)

    matplotlib.pyplot.plot(accs_real, c="green")
    matplotlib.pyplot.plot(accs_fake, c="red")
    matplotlib.pyplot.savefig("progress/acuracy.png")
    matplotlib.pyplot.close()

if __name__ == '__main__':
    latent_dims = 5
    domain_dims = 2
    n_epochs = 15000 + 1 # additional epoch to collect statistics before exit
    n_batches = 128
    eval_freq = 100

    d = define_discriminator(domain_dims)
    g = define_generator(latent_dims, domain_dims)
    gan = define_gan(d, g, latent_dims)

    train(d, g, gan, latent_dims, n_epochs, n_batches, eval_freq)
