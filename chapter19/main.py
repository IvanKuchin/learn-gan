import datetime
import os.path

import numpy as np
import tensorflow as tf
from matplotlib import pyplot


def save_images(images, name):
    cardinality = len(images.shape)
    if cardinality != 4:
        print(f"images cardinality expected to be 4, but given shape is {images.shape}")
        return

    edge = int(np.sqrt(images.shape[0]))
    min, max = np.min(images), np.max(images)
    images = (images - min) / (max - min)

    for i in range(edge**2):
        pyplot.subplot(edge, edge, i+1)
        pyplot.axis(False)
        pyplot.imshow(images[i], cmap="gray_r")
    pyplot.savefig(f"progress/{name}.png")
    pyplot.close()


def load_dataset():
    (trainX, labels), _ = tf.keras.datasets.fashion_mnist.load_data()
    trainX = trainX / 255.0 * 2.0 - 1
    trainX = np.expand_dims(trainX, -1)
    labels = np.expand_dims(labels, -1)
    return trainX, labels


def prep():
    if not os.path.isdir("progress"):
        os.mkdir("progress")


def define_discriminator(input_shape):
    init = tf.keras.initializers.RandomNormal(stddev=0.02)

    inputs = tf.keras.Input(input_shape, name="images")

    conv1 = tf.keras.layers.Conv2D(32, (3, 3), strides=(2, 2), padding="same", kernel_initializer=init)(inputs)
    bn1 = tf.keras.layers.BatchNormalization()(conv1)
    act1 = tf.keras.layers.LeakyReLU(alpha=0.2)(bn1)

    conv2 = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding="same", kernel_initializer=init)(act1)
    bn2 = tf.keras.layers.BatchNormalization()(conv2)
    act2 = tf.keras.layers.LeakyReLU(alpha=0.2)(bn2)

    conv3 = tf.keras.layers.Conv2D(96, (3, 3), strides=(2, 2), padding="same", kernel_initializer=init)(act2)
    bn3 = tf.keras.layers.BatchNormalization()(conv3)
    act3 = tf.keras.layers.LeakyReLU(alpha=0.2)(bn3)

    conv4 = tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding="same", kernel_initializer=init)(act3)
    bn4 = tf.keras.layers.BatchNormalization()(conv4)
    act4 = tf.keras.layers.LeakyReLU(alpha=0.2)(bn4)

    flat = tf.keras.layers.Flatten()(act4)
    drop1 = tf.keras.layers.Dropout(0.4)(flat)
    fakeness = tf.keras.layers.Dense(1, activation="sigmoid", name="fakeness")(drop1)

    drop2 = tf.keras.layers.Dropout(0.4)(flat)
    category = tf.keras.layers.Dense(10, activation="softmax", name="category")(drop2)

    model = tf.keras.Model(inputs=inputs, outputs=[fakeness, category], name="discriminator")

    opt = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss=["binary_crossentropy", "sparse_categorical_crossentropy"], optimizer=opt)

    return model


def define_generator(latent_dims):
    init = tf.keras.initializers.RandomNormal(stddev=0.02)

    latent = tf.keras.Input(latent_dims, name="latent")
    dense1 = tf.keras.layers.Dense(7*7*128, kernel_initializer=init)(latent)
    act1 = tf.keras.layers.LeakyReLU(alpha=0.2)(dense1)
    reshape1 = tf.keras.layers.Reshape((7, 7, 128))(act1)

    category = tf.keras.Input(1, name="category")
    emb = tf.keras.layers.Embedding(10, 50)(category)
    dense2 = tf.keras.layers.Dense(7*7, kernel_initializer=init)(emb)
    reshape2 = tf.keras.layers.Reshape((7, 7, 1))(dense2)

    concat = tf.keras.layers.Concatenate()([reshape1, reshape2])

    conv3tr = tf.keras.layers.Conv2DTranspose(96, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init)(concat)
    bn3 = tf.keras.layers.BatchNormalization()(conv3tr)
    act3 = tf.keras.layers.LeakyReLU(alpha=0.2)(bn3)

    conv4tr = tf.keras.layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init)(act3)
    bn4 = tf.keras.layers.BatchNormalization()(conv4tr)
    act4 = tf.keras.layers.LeakyReLU(alpha=0.2)(bn4)

    conv5 = tf.keras.layers.Conv2D(1, (3, 3), strides=(1, 1), padding="same", kernel_initializer=init, activation="tanh", name="images")(act4)

    model = tf.keras.Model(inputs=[latent, category], outputs=conv5, name="generator")
    return model


def define_gan(gen, disc):
    for layer in disc.layers:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    latent, category_in = gen.inputs

    gen_out = gen([latent, category_in])
    fakeness, category_out = disc(gen_out)

    model = tf.keras.Model(inputs=[latent, category_in], outputs=[fakeness, category_out], name="gan")
    opt = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss=["binary_crossentropy", "sparse_categorical_crossentropy"], optimizer=opt)

    return model


def define_writer():
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return tf.summary.create_file_writer(f"logs/{ts}")


def generate_real_samples(ds, batch):
    ix = np.random.randint(0, ds[0].shape[0], batch)
    X = ds[0][ix]
    labels = ds[1][ix]
    y = np.ones((batch, 1))
    return X, labels, y


def generate_latent_points(batch, latent_dims):
    points = np.random.randn(batch * latent_dims)
    points = np.reshape(points, (batch, latent_dims))
    labels = np.random.randint(0, 10, batch)
    return points, labels


def generate_fake_samples(gen, batch):
    latent_dims = gen.input[0].shape[-1]
    points, labels = generate_latent_points(batch, latent_dims)
    X = gen.predict([points, labels], verbose=0)
    y = np.zeros((batch, 1))
    return X, labels, y


def train_discriminator(gen, disc, batch, dataset):
    X_real, labels_real, y_real = generate_real_samples(dataset, batch // 2)
    X_fake, labels_fake, y_fake = generate_fake_samples(gen, batch // 2)
    _, disc_loss_real_fakeness, disc_loss_real_category = disc.train_on_batch(X_real, [y_real, labels_real])
    _, disc_loss_fake_fakeness, disc_loss_fake_category = disc.train_on_batch(X_fake, [y_fake, labels_fake])
    return disc_loss_real_fakeness, disc_loss_real_category, disc_loss_fake_fakeness, disc_loss_fake_category


def train_gan(gen, gan, batch):
    latent_dims = gen.input[0].shape[-1]
    points, labels = generate_latent_points(batch, latent_dims)
    y = np.ones((batch, 1))
    _, gan_loss_fakeness, gan_loss_category = gan.train_on_batch([points, labels], [y, labels])
    return gan_loss_fakeness, gan_loss_category


def train(gen, disc, gan, cfg):
    writer = define_writer()
    steps_per_epoch = cfg["dataset"][0].shape[0] // cfg["batch"]

    for epoch in range(cfg["epochs"]):
        list_disc_loss_real_fakeness = list()
        list_disc_loss_real_category = list()
        list_disc_loss_fake_fakeness = list()
        list_disc_loss_fake_category = list()
        list_gan_loss_fakeness = list()
        list_gan_loss_category = list()
        for step in range(steps_per_epoch):
            disc_loss_real_fakeness, disc_loss_real_category, disc_loss_fake_fakeness, disc_loss_fake_category = train_discriminator(gen, disc, cfg["batch"], cfg["dataset"])
            gan_loss_fakeness, gan_loss_category = train_gan(gen, gan, cfg["batch"])

            print(f"{epoch} {step:3d}/{steps_per_epoch}: disc_loss_real {disc_loss_real_fakeness:.3f}/{disc_loss_real_category:.3f},  disc_loss_fake {disc_loss_fake_fakeness:.3f}/{disc_loss_fake_category:.3f}, gan {gan_loss_fakeness:.3f}/{gan_loss_category:.3f}")

            list_disc_loss_real_fakeness.append(disc_loss_real_fakeness)
            list_disc_loss_real_category.append(disc_loss_real_category)
            list_disc_loss_fake_fakeness.append(disc_loss_fake_fakeness)
            list_disc_loss_fake_category.append(disc_loss_fake_category)
            list_gan_loss_fakeness.append(gan_loss_fakeness)
            list_gan_loss_category.append(gan_loss_category)

        summarize_performance(gen, epoch)
        with writer.as_default():
            tf.summary.scalar("list_disc_loss_real_fakeness", np.mean(list_disc_loss_real_fakeness), step=epoch)
            tf.summary.scalar("list_disc_loss_real_category", np.mean(list_disc_loss_real_category), step=epoch)
            tf.summary.scalar("list_disc_loss_fake_fakeness", np.mean(list_disc_loss_fake_fakeness), step=epoch)
            tf.summary.scalar("list_disc_loss_fake_category", np.mean(list_disc_loss_fake_category), step=epoch)
            tf.summary.scalar("list_gan_loss_fakeness", np.mean(list_gan_loss_fakeness), step=epoch)
            tf.summary.scalar("list_gan_loss_category", np.mean(list_gan_loss_category), step=epoch)


def summarize_performance(gen, epoch):
    latent_dims = gen.input[0].shape[-1]
    points, _ = generate_latent_points(100, latent_dims)
    labels = np.array([i for i in range(10) for _ in range(10)])
    images = gen.predict([points, labels])
    save_images(images, f"{epoch:03d}")
    return


def main():
    cfg = {
        "batch": 64,
        "latent_dims": 100,
        "epochs": 100,
        "categories": 10,
        "dataset": load_dataset()
    }

    prep()
    # save_images(cfg["dataset"][0][:25], "init")
    disc = define_discriminator(cfg["dataset"][0].shape[1:])
    gen = define_generator(cfg["latent_dims"])
    gan = define_gan(gen, disc)
    # disc.summary()
    # gen.summary()
    # gan.summary()
    # tf.keras.utils.plot_model(disc, "progress/disc.png", show_shapes=True)
    # tf.keras.utils.plot_model(gen, "progress/gen.png", show_shapes=True)
    # tf.keras.utils.plot_model(gan, "progress/gan.png", show_shapes=True)

    train(gen, disc, gan, cfg)

    return

if __name__ == "__main__":
    main()
