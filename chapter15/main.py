import datetime

import numpy as np
import tensorflow as tf
from matplotlib import pyplot


def load_dataset():
    (trainX, _), (_, _) = tf.keras.datasets.mnist.load_data()
    trainX = trainX / 255.0 * 2 - 1
    trainX = np.expand_dims(trainX, -1)
    return trainX


def define_discriminator(cfg):
    init = tf.keras.initializers.RandomNormal(stddev=0.02)
    inputs = tf.keras.Input(shape=cfg["in_shape"])

    # 14x14
    conv1 = tf.keras.layers.Conv2D(64, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init)(inputs)
    bn1 = tf.keras.layers.BatchNormalization()(conv1)
    act1 = tf.keras.layers.LeakyReLU(alpha=0.2)(bn1)

    # 7x7
    conv2 = tf.keras.layers.Conv2D(128, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init)(act1)
    bn2 = tf.keras.layers.BatchNormalization()(conv2)
    act2 = tf.keras.layers.LeakyReLU(alpha=0.2)(bn2)

    # classifier
    flat1 = tf.keras.layers.Flatten()(act2)
    dens1 = tf.keras.layers.Dense(1, activation="linear", kernel_initializer=init)(flat1)

    model = tf.keras.Model(inputs=inputs, outputs=dens1, name="discriminator")
    opt = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss="mse", optimizer=opt, metrics=["accuracy"])

    return model


def define_generator(latent_dims):
    init = tf.keras.initializers.RandomNormal(stddev=0.02)

    inputs = tf.keras.Input(shape=latent_dims)

    dens1 = tf.keras.layers.Dense(7*7*256, kernel_initializer=init)(inputs)
    resh1 = tf.keras.layers.Reshape((7, 7, 256))(dens1)
    act1 = tf.keras.layers.LeakyReLU(alpha=0.2)(resh1)

    # 14x14
    conv2tr = tf.keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init)(act1)
    bn2 = tf.keras.layers.BatchNormalization()(conv2tr)
    act2 = tf.keras.layers.LeakyReLU(alpha=0.2)(bn2)

    # 28x28
    conv3tr = tf.keras.layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init)(act2)
    bn3 = tf.keras.layers.BatchNormalization()(conv3tr)
    act3 = tf.keras.layers.LeakyReLU(alpha=0.2)(bn3)

    # 28x28
    conv4 = tf.keras.layers.Conv2D(1, (7, 7), strides=(1, 1), padding="same", kernel_initializer=init, activation="tanh")(act3)

    model = tf.keras.Model(inputs=inputs, outputs=conv4, name="generator")

    return model


def define_gan(gen, disc):
    for layer in disc.layers:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    model = tf.keras.models.Sequential()
    model.add(gen)
    model.add(disc)

    opt = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss="mse", optimizer=opt, metrics=["accuracy"])

    return model


def generate_latent_points(batch, latent_dims):
    points = np.random.randn(batch * latent_dims)
    points = np.reshape(points, (batch, latent_dims))
    return points


def generate_fake_samples(gen, batch):
    latent_dims = gen.input.shape[1]
    latent_points = generate_latent_points(batch, latent_dims)
    x = gen.predict(latent_points, verbose=0)
    y = np.zeros((batch, 1))
    return x, y


def generate_real_samples(ds, batch):
    ix = np.random.randint(0, ds.shape[0], batch)
    x = ds[ix]
    y = np.ones((batch, 1))
    return x, y


def train_disc(gen, disc, cfg):
    realX, realY = generate_real_samples(cfg["dataset"], cfg["batch"] // 2)
    fakeX, fakeY = generate_fake_samples(gen, cfg["batch"] // 2)

    loss_real, acc_real = disc.train_on_batch(realX, realY)
    loss_fake, acc_fake = disc.train_on_batch(fakeX, fakeY)

    return loss_real, acc_real, loss_fake, acc_fake


def train_gan(gan, gen, cfg):
    fakeX = generate_latent_points(cfg["batch"], cfg["latent_dims"])
    realY = np.ones((cfg["batch"], 1))
    loss, acc = gan.train_on_batch(fakeX, realY)
    return loss, acc


def create_writer():
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = tf.summary.create_file_writer(f"logs/{ts}")
    return writer


def train(gen, disc, gan, cfg):
    writer = create_writer()
    ds = cfg["dataset"]
    steps_per_epoch = ds.shape[0] // cfg["batch"]
    for epoch in range(cfg["epochs"]):
        for step in range(steps_per_epoch):
            # for step in range(5):
            loss_real, acc_real, loss_fake, acc_fake = train_disc(gen, disc, cfg)
            gen_loss, gen_acc = train_gan(gan, gen, cfg)
            print(f"{epoch}, {step}/{steps_per_epoch}: d_{loss_real=:.3f}, d_{loss_fake=:.3f}, g_{gen_loss=:.3f}, d_{acc_real=:.3f}, d_{acc_fake=:.3f}")

        d_loss_real, d_acc_real, d_loss_fake, d_acc_fake, g_loss, g_acc = summarize_performance(gen, disc, gan, cfg, epoch)
        with writer.as_default():
            tf.summary.scalar("d_loss_real", d_loss_real, step=epoch)
            tf.summary.scalar("d_loss_fake", d_loss_fake, step=epoch)
            tf.summary.scalar("d_acc_real", d_acc_real, step=epoch)
            tf.summary.scalar("d_acc_fake", d_acc_fake, step=epoch)
            tf.summary.scalar("g_loss", g_loss, step=epoch)
            tf.summary.scalar("g_acc", g_acc, step=epoch)

def summarize_performance(gen, disc, gan, cfg, epoch):
    latent_points = generate_latent_points(cfg["batch"], cfg["latent_dims"])

    fakeX, fakeY = generate_fake_samples(gen, cfg["batch"])
    realX, realY = generate_real_samples(cfg["dataset"], cfg["batch"])

    d_loss_real, d_acc_real = disc.evaluate(realX, realY)
    d_loss_fake, d_acc_fake = disc.evaluate(fakeX, fakeY)

    g_loss, g_acc = gan.evaluate(latent_points, realY)

    save_images(fakeX[:25], f"{epoch:03d}")

    return d_loss_real, d_acc_real, d_loss_fake, d_acc_fake, g_loss, g_acc

def save_images(images, name):
    cardinality = len(images.shape)
    if cardinality != 4:
        print("image cardinality must be 4, but given ", images.shape)
        return None

    sqrt = int(np.sqrt(images.shape[0]))
    for i in range(sqrt**2):
        pyplot.subplot(sqrt, sqrt, i+1)
        pyplot.axis(False)
        pyplot.imshow(images[i], cmap="gray_r")
    pyplot.savefig(f"progress/{name}.png")
    pyplot.close()


def main():
    trainX = load_dataset()
    # print(f"{trainX.shape=}")
    save_images(trainX[:25], "init")

    cfg = {"epochs": 100, "batch": 64, "in_shape": trainX.shape[1:], "latent_dims": 100, "dataset": trainX}

    d = define_discriminator(cfg)
    # d.summary()
    g = define_generator(cfg["latent_dims"])
    # g.summary()
    gan = define_gan(g, d)
    # gan.summary(show_trainable=True, expand_nested=True)

    train(g, d, gan, cfg)

if __name__ == '__main__':
    main()
