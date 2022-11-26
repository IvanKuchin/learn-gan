import datetime
import os.path

import tensorflow as tf
import numpy as np
from matplotlib import pyplot


def define_discriminator(input_shape):
    inputs = tf.keras.Input(input_shape)

    # 14x14
    conv1 = tf.keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same")(inputs)
    act1 = tf.keras.layers.LeakyReLU(alpha=0.2)(conv1)

    # 7x7
    conv2 = tf.keras.layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same")(act1)
    act2 = tf.keras.layers.LeakyReLU(alpha=0.2)(conv2)

    # classification
    fl = tf.keras.layers.Flatten()(act2)
    dr = tf.keras.layers.Dropout(0.4)(fl)
    classification = tf.keras.layers.Dense(1, activation="sigmoid")(dr)

    model = tf.keras.Model(inputs = inputs, outputs=classification, name="discriminator")
    opt = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

    return model


def define_generator(latent_dims):
    inputs = tf.keras.Input(latent_dims)

    feature_map = 7*7
    dense1 = tf.keras.layers.Dense(feature_map*128)(inputs)
    act1 = tf.keras.layers.LeakyReLU(alpha=0.2)(dense1)
    reshape1 = tf.keras.layers.Reshape((7, 7, 128))(act1)

    # 14x14
    conv2tr = tf.keras.layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding="same")(reshape1)
    act2 = tf.keras.layers.LeakyReLU(alpha=0.2)(conv2tr)

    # 28x28
    conv3tr = tf.keras.layers.Conv2DTranspose(32, (4, 4), strides=(2, 2), padding="same")(act2)
    act3 = tf.keras.layers.LeakyReLU(alpha=0.2)(conv3tr)

    conv4 = tf.keras.layers.Conv2D(1, (1, 1), strides=(1, 1), padding="same", activation="tanh")(act3)

    model = tf.keras.Model(inputs=inputs, outputs=conv4, name="generator")
    return model


def define_gan(gen, disc):
    disc.trainable = False

    # inputs = tf.keras.Input(gen.input.shape[1:])
    inputs = gen.input
    gen_out = gen(inputs)
    disc_out = disc(gen_out)

    model = tf.keras.Model(inputs=inputs, outputs=disc_out, name="gan")
    opt = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

    return model


def display_image(images, name):
    if not os.path.isdir("progress"):
        os.mkdir("progress")

    cardinality = len(images.shape)
    if cardinality != 4:
        print(f"expected cardinality 4, but given {images.shape}")
        return

    side = int(np.sqrt(images.shape[0]))
    for i in range(side**2):
        pyplot.subplot(side, side, i+1)
        pyplot.axis(False)
        pyplot.imshow(images[i], cmap="gray_r")
    pyplot.savefig(f"progress/{name}.png")
    pyplot.close()


def load_dataset():
    (trainX, _), (_, _) = tf.keras.datasets.fashion_mnist.load_data()
    trainX = trainX / 255.0 * 2 - 1
    trainX = np.expand_dims(trainX, -1)
    return trainX


def define_writer():
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = f"logs/{ts}"
    writer = tf.summary.create_file_writer(logdir)
    return writer


def generate_real_samples(ds, batch):
    ix = np.random.randint(0, ds.shape[0], batch)
    x = ds[ix]
    y = np.ones((batch, 1))
    return x, y


def generate_latent_points(batch, latent_dims):
    r = np.random.randn(batch * latent_dims)
    r = np.reshape(r, (batch, latent_dims))
    return r


def generate_fake_samples(gen, batch):
    latent_points = generate_latent_points(batch, gen.input.shape[-1])
    x = gen(latent_points)
    y = np.zeros((batch, 1))
    return x, y


def train_discriminator(gen, disc, ds, batch):
    realX, realY = generate_real_samples(ds, batch // 2)
    fakeX, fakeY = generate_fake_samples(gen, batch // 2)

    loss_real, acc_real = disc.train_on_batch(realX, realY)
    loss_fake, acc_fake = disc.train_on_batch(fakeX, fakeY)
    return loss_real, loss_fake, acc_real, acc_fake


def train_gen(gen, gan, batch):
    latent_dims = gen.input.shape[-1]
    latent_points = generate_latent_points(batch, latent_dims)
    y = np.ones((batch, 1))
    loss, acc = gan.train_on_batch(latent_points, y)
    return loss, acc


def train(gen, disc, gan, cfg):
    writer = define_writer()
    batch = cfg["batch"]

    steps_per_epoch = cfg["dataset"].shape[0] // batch

    for epoch in range(cfg["epochs"]):
        losses_real, losses_fake, accs_real, accs_fake, losses_gen, accs_gen = list(), list(), list(), list(), list(), list()
        for step in range(steps_per_epoch):
            loss_real, loss_fake, acc_real, acc_fake = train_discriminator(gen, disc, cfg["dataset"], batch)
            loss_gen, acc_gen = train_gen(gen, gan, batch)

            print(f"{epoch} {step:3d}/{steps_per_epoch}: {loss_real=:.3f}, {loss_fake=:.3f}, {loss_gen=:.3f}, {acc_real=:.2f}, {acc_fake=:.2f}, {acc_gen=:.2f}")
            losses_real.append(loss_real)
            losses_fake.append(loss_fake)
            accs_real.append(acc_real)
            accs_fake.append(acc_fake)
            losses_gen.append(loss_gen)
            accs_gen.append(acc_gen)

        with writer.as_default():
            tf.summary.scalar("loss_real", np.mean(losses_real), step=epoch)
            tf.summary.scalar("loss_fake", np.mean(losses_fake), step=epoch)
            tf.summary.scalar("acc_real", np.mean(accs_real), step=epoch)
            tf.summary.scalar("acc_fake", np.mean(accs_fake), step=epoch)
            tf.summary.scalar("loss_gen", np.mean(losses_gen), step=epoch)
            tf.summary.scalar("acc_gen", np.mean(accs_gen), step=epoch)

        summarize_performance(gen, batch=25, name=f"{epoch:03d}")
    return


def summarize_performance(gen, batch, name):
    latent_dims = gen.input.shape[-1]
    latent_points = generate_latent_points(batch, latent_dims)
    images = gen.predict(latent_points)
    display_image(images, name)
    return


def main():
    cfg = {
        "epochs": 100,
        "latent_dims": 100,
        "batch": 128,
        "dataset": load_dataset(),
    }

    # display_image(cfg["dataset"][:25], "init")
    disc = define_discriminator(cfg["dataset"].shape[1:])
    gen = define_generator(cfg["latent_dims"])
    gan = define_gan(gen, disc)
    # tf.keras.utils.plot_model(disc, "progress/disc.png", show_shapes=True)
    # tf.keras.utils.plot_model(gen, "progress/gen.png", show_shapes=True)
    # tf.keras.utils.plot_model(gan, "progress/gan.png", show_shapes=True)
    train(gen, disc, gan, cfg)
    return


if __name__ == "__main__":
    main()
