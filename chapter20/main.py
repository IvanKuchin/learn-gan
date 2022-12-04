import datetime
import os.path

import tensorflow as tf
import numpy as np
from matplotlib import pyplot


def activation(x):
    exp = tf.exp(x)
    _sum = tf.reduce_sum(exp, axis=-1, keepdims=True)
    return _sum / (_sum + 1)


def load_unsupervised_dataset():
    (x, _), (_, _) = tf.keras.datasets.mnist.load_data()
    x = x / 255.0 * 2.0 - 1
    x = np.expand_dims(x, -1)
    return x


def load_supervised_dataset(n_samples, n_classes):
    (images, _labels), (_, _) = tf.keras.datasets.mnist.load_data()
    images = images / 255.0 * 2.0 - 1
    images = np.expand_dims(images, -1)

    samples_per_class = n_samples // n_classes

    x_list, y_list = list(), list()
    for c in range(n_classes):
        x_class = images[_labels == c]
        ix = np.random.randint(0, x_class.shape[0], samples_per_class)
        x_select = x_class[ix]
        x_list.append(x_select)
        y_list.append(np.ones((samples_per_class, 1)) * c)

    x, y = np.asarray(x_list), np.asarray(y_list)
    x = x.reshape((-1,) + x.shape[2:])
    y = y.reshape((-1,) + y.shape[2:])
    return x, y


def generate_latent_points(batch, latent_dims):
    noise = np.random.randn(batch*latent_dims)
    noise = np.reshape(noise, (batch, latent_dims))
    return noise


def generate_fake_samples(gen, batch):
    latent_dims = gen.input.shape[-1]
    noise = generate_latent_points(batch, latent_dims)
    x = gen.predict(noise, verbose=0)
    y = np.zeros((batch, 1))
    return x, y


def generate_real_samples(ds, batch):
    ix = np.random.randint(0, ds.shape[0], batch)
    x = ds[ix]
    y = np.ones((batch, 1))
    return x, y


def save_images(images, name):
    cardinality = len(images.shape)
    if cardinality != 4:
        print(f"expected cardinality is 4, but given shape is {images.shape}")
        return

    min, max = images.min(), images.max()
    images = (images - min) / (max - min)

    edge = int(np.sqrt(images.shape[0]))
    for i in range(edge**2):
        pyplot.subplot(edge, edge, i+1)
        pyplot.axis(False)
        pyplot.imshow(images[i], cmap="gray_r")
    pyplot.savefig(f"progress/{name}.png")
    pyplot.close()

def prep():
    if not os.path.isdir("progress"):
        os.mkdir("progress")


def define_discriminators(image_shape):
    init = tf.keras.initializers.RandomNormal(stddev=0.02)
    inputs = tf.keras.Input(image_shape, name="images")

    conv1 = tf.keras.layers.Conv2D(32, (3, 3), strides=(2, 2), padding="same", kernel_initializer=init)(inputs)
    bn1 = tf.keras.layers.BatchNormalization()(conv1)
    act1 = tf.keras.layers.LeakyReLU(alpha=0.2)(bn1)

    conv2 = tf.keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same", kernel_initializer=init)(act1)
    bn2 = tf.keras.layers.BatchNormalization()(conv2)
    act2 = tf.keras.layers.LeakyReLU(alpha=0.2)(bn2)

    conv3 = tf.keras.layers.Conv2D(96, (3, 3), strides=(2, 2), padding="same", kernel_initializer=init)(act2)
    bn3 = tf.keras.layers.BatchNormalization()(conv3)
    act3 = tf.keras.layers.LeakyReLU(alpha=0.2)(bn3)

    fl = tf.keras.layers.Flatten()(act3)
    dr = tf.keras.layers.Dropout(0.4)(fl)
    dense1 = tf.keras.layers.Dense(10)(dr)

    softmax = tf.keras.activations.softmax(dense1)
    disc_s = tf.keras.Model(inputs=inputs, outputs=softmax, name="disc_supervised")
    disc_s.compile(loss="sparse_categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5))

    custom_act = tf.keras.layers.Lambda(activation)(dense1)
    disc_u = tf.keras.Model(inputs=inputs, outputs=custom_act, name="disc_unsupervised")
    disc_u.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5), metrics=["accuracy"])

    return disc_u, disc_s


def define_generator(latent_dims):
    inputs = tf.keras.Input(latent_dims, name="noise")
    dense1 = tf.keras.layers.Dense(7*7*128)(inputs)
    bn1 = tf.keras.layers.BatchNormalization()(dense1)
    act1 = tf.keras.layers.LeakyReLU(alpha=0.2)(bn1)
    reshape1 = tf.keras.layers.Reshape((7, 7, 128))(act1)

    conv2tr = tf.keras.layers.Conv2DTranspose(96, (4, 4), strides=(2, 2), padding="same")(reshape1)
    bn2 = tf.keras.layers.BatchNormalization()(conv2tr)
    act2 = tf.keras.layers.LeakyReLU(alpha=0.2)(bn2)

    conv3tr = tf.keras.layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding="same")(act2)
    bn2 = tf.keras.layers.BatchNormalization()(conv3tr)
    act3 = tf.keras.layers.LeakyReLU(alpha=0.2)(bn2)

    conv4 = tf.keras.layers.Conv2D(1, (3, 3), strides=(1, 1), padding="same", activation="tanh")(act3)

    model = tf.keras.Model(inputs=inputs, outputs=conv4, name="generator")
    return model


def define_gan(gen, disc_u):
    for layer in disc_u.layers:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    inputs = gen.input
    gen_out = gen(inputs)
    disc_u_out = disc_u(gen_out)

    gan = tf.keras.Model(inputs=inputs, outputs=disc_u_out, name="gan")
    opt = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    gan.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

    return gan


def train_discriminator_supervised(disc, ds, batch):
    _x, _labels = ds
    ix = np.random.randint(0, _x.shape[0], batch)
    x, labels = _x[ix], _labels[ix]

    loss = disc.train_on_batch(x, labels)
    return loss


def train_discriminator_unsupervised(gen, disc, ds, batch):
    realX, realy = generate_real_samples(ds, batch // 2)
    fakeX, fakey = generate_fake_samples(gen, batch // 2)
    loss_real, acc_real = disc.train_on_batch(realX, realy)
    loss_fake, acc_fake = disc.train_on_batch(fakeX, fakey)
    return loss_real, loss_fake, acc_real, acc_fake


def train_gan(gen, gan, batch):
    latent_dims = gen.input.shape[-1]
    noise = generate_latent_points(batch, latent_dims)
    y = np.ones((batch, 1))
    loss, acc = gan.train_on_batch(noise, y)
    return loss, acc


def train(gen, disc_u, disc_s, gan, cfg):
    writer = define_writer()
    batch = cfg["batch"]
    steps_per_epoch = cfg["dataset_unsupervised"].shape[0] // batch
    for epoch in range(cfg["epochs"]):
        list_loss_disc_s = list()
        list_loss_disc_u_real = list()
        list_loss_disc_u_fake = list()
        list_acc_disc_u_real = list()
        list_acc_disc_u_fake = list()
        list_loss_gan = list()
        list_acc_gan = list()
        for i in range(steps_per_epoch):
            loss_disc_s = train_discriminator_supervised(disc_s, cfg["dataset_supervised"], batch // 3)
            loss_disc_u_real, loss_disc_u_fake, acc_disc_u_real, acc_disc_u_fake = train_discriminator_unsupervised(gen, disc_u, cfg["dataset_unsupervised"], batch // 3 * 2)
            loss_gan, acc_gan = train_gan(gen, gan, batch)

            print(f"{epoch} {i:3d}/{steps_per_epoch}: loss disc_s/disc_u_real/disc_u_fake/gan {loss_disc_s:.3f}/{loss_disc_u_real:.3f}/{loss_disc_u_fake:.3f}/{loss_gan:.3f}")

            list_loss_disc_s.append(loss_disc_s)
            list_loss_disc_u_real.append(loss_disc_u_real)
            list_loss_disc_u_fake.append(loss_disc_u_fake)
            list_acc_disc_u_real.append(acc_disc_u_real)
            list_acc_disc_u_fake.append(acc_disc_u_fake)
            list_loss_gan.append(loss_gan)
            list_acc_gan.append(acc_gan)

        with writer.as_default():
            tf.summary.scalar("loss_disc_s", np.mean(list_loss_disc_s), step=epoch)
            tf.summary.scalar("loss_disc_u_real", np.mean(list_loss_disc_u_real), step=epoch)
            tf.summary.scalar("loss_disc_u_fake", np.mean(list_loss_disc_u_fake), step=epoch)
            tf.summary.scalar("acc_disc_u_real", np.mean(list_acc_disc_u_real), step=epoch)
            tf.summary.scalar("acc_disc_u_fake", np.mean(list_acc_disc_u_fake), step=epoch)
            tf.summary.scalar("loss_gan", np.mean(list_loss_gan), step=epoch)
            tf.summary.scalar("acc_gan", np.mean(list_acc_gan), step=epoch)
        summarize_performance(gen, 100, epoch)
    return


def summarize_performance(gen, batch, epoch):
    latent_dims = gen.input.shape[-1]
    noise = generate_latent_points(batch, latent_dims)
    images = gen.predict(noise, verbose=0)
    save_images(images, f"{epoch}")


def define_writer():
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return tf.summary.create_file_writer(f"logs/{ts}")


def main():
    prep()

    cfg = {
        "batch": 100,
        "epochs": 50,
        "latent_dims": 100,
        "dataset_unsupervised": load_unsupervised_dataset(),
        "dataset_supervised": load_supervised_dataset(n_samples=100, n_classes=10)
    }

    # print(cfg["dataset_unsupervised"].shape)
    # print(cfg["dataset_supervised"][0].shape)
    # print(cfg["dataset_supervised"][1].shape)
    # save_images(cfg["dataset_supervised"][0], "ds_supervised")
    # save_images(cfg["dataset_unsupervised"][:100], "ds_unsupervised")

    disc_u, disc_s = define_discriminators(cfg["dataset_unsupervised"].shape[1:])
    gen = define_generator(cfg["latent_dims"])
    gan = define_gan(gen, disc_u)
    # tf.keras.utils.plot_model(disc_u, "progress/disc_u.png", show_shapes=True)
    # tf.keras.utils.plot_model(disc_s, "progress/disc_s.png", show_shapes=True)
    # tf.keras.utils.plot_model(gen, "progress/gen.png", show_shapes=True)
    # tf.keras.utils.plot_model(gan, "progress/gan.png", show_shapes=True)

    train(gen, disc_u, disc_s, gan, cfg)

    return


if __name__ == "__main__":
    main()

