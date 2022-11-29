import datetime
import os.path

import numpy as np
import tensorflow as tf
from matplotlib import pyplot


def save_images(images, name):
    cardinality = len(images.shape)
    if cardinality != 4:
        print(f"images cardinality must be 4, but given {images.shape}")

    min, max = np.min(images), np.max(images)
    images = (images - min) / (max - min)
    sqrt = int(np.sqrt(images.shape[0]))

    for i in range(sqrt**2):
        pyplot.subplot(sqrt, sqrt, i+1)
        pyplot.axis(False)
        pyplot.imshow(images[i], cmap="gray_r")
    pyplot.savefig(f"progress/{name}.png")
    pyplot.close()


def load_dataset():
    (trainX, _), (_, _) = tf.keras.datasets.mnist.load_data()
    trainX = trainX / 255.0 * 2 - 1
    trainX = np.expand_dims(trainX, -1)
    return trainX


def define_discriminator(input_shape, cat_dims):
    init = tf.keras.initializers.RandomNormal(stddev=0.02)

    inputs = tf.keras.Input(input_shape)

    conv1 = tf.keras.layers.Conv2D(64, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init)(inputs)
    bn1 = tf.keras.layers.BatchNormalization()(conv1)
    act1 = tf.keras.layers.LeakyReLU(alpha=0.2)(bn1)

    conv2 = tf.keras.layers.Conv2D(128, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init)(act1)
    bn2 = tf.keras.layers.BatchNormalization()(conv2)
    act2 = tf.keras.layers.LeakyReLU(alpha=0.2)(bn2)

    conv3 = tf.keras.layers.Conv2D(256, (4, 4), strides=(1, 1), padding="same", kernel_initializer=init)(act2)
    bn3 = tf.keras.layers.BatchNormalization()(conv3)
    act3 = tf.keras.layers.LeakyReLU(alpha=0.2)(bn3)

    flat = tf.keras.layers.Flatten()(act3)
    out_disc = tf.keras.layers.Dense(1, activation="sigmoid")(flat)

    disc = tf.keras.Model(inputs=inputs, outputs=out_disc, name="discriminator")
    opt = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    disc.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

    dense2 = tf.keras.layers.Dense(128)(flat)
    bn4 = tf.keras.layers.BatchNormalization()(dense2)
    act4 = tf.keras.layers.LeakyReLU(alpha=0.2)(bn4)

    out_aux = tf.keras.layers.Dense(cat_dims, activation="softmax")(act4)

    aux = tf.keras.Model(inputs=inputs, outputs=out_aux, name="aux")

    return disc, aux


def define_generator(latent_dims, cat_dims):
    init = tf.keras.initializers.RandomNormal(stddev=0.02)

    inputs = tf.keras.Input((latent_dims + cat_dims))
    dense1 = tf.keras.layers.Dense(7*7*256)(inputs)
    bn1 = tf.keras.layers.BatchNormalization()(dense1)
    act1 = tf.keras.layers.LeakyReLU(alpha=0.2)(bn1)
    reshape1 = tf.keras.layers.Reshape((7, 7, 256))(act1)

    conv2tr = tf.keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init)(reshape1)
    bn2 = tf.keras.layers.BatchNormalization()(conv2tr)
    act2 = tf.keras.layers.LeakyReLU(alpha=0.2)(bn2)

    conv3tr = tf.keras.layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init)(act2)
    bn3 = tf.keras.layers.BatchNormalization()(conv3tr)
    act3 = tf.keras.layers.LeakyReLU(alpha=0.2)(bn3)

    conv4 = tf.keras.layers.Conv2D(1, (4, 4), strides=(1, 1), padding="same", kernel_initializer=init, activation="tanh")(act3)

    gen = tf.keras.Model(inputs=inputs, outputs=conv4, name="generator")

    return gen


def define_gan(gen, disc, aux):
    for layer in disc.layers:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    inputs = gen.input

    gen_out = gen(inputs)
    disc_out = disc(gen_out)
    aux_out = aux(gen_out)

    model = tf.keras.Model(inputs=inputs, outputs=[disc_out, aux_out], name="gan")
    opt = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss=["binary_crossentropy", "categorical_crossentropy"], optimizer=opt)

    return model


def define_writer():
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = tf.summary.create_file_writer(f"logs/{ts}")
    return writer


def generate_real_samples(ds, batch):
    ix = np.random.randint(0, ds.shape[0], batch)
    x = ds[ix]
    y = np.ones((batch, 1))
    return x, y


def generate_latent_points(batch, latent_dims, cat_dims):
    latent_points = np.random.randn(batch * latent_dims)
    latent_points = np.reshape(latent_points, (batch, latent_dims))

    categories = tf.one_hot(np.random.randint(0, cat_dims, batch), cat_dims)
    latent_and_cats = np.hstack([latent_points, categories.numpy()])
    return latent_and_cats, latent_points, categories


def generate_fake_samples(gen, latent_dims, batch):
    input_length = gen.input.shape[-1]
    cat_dims = input_length - latent_dims

    latent_points, _, _ = generate_latent_points(batch, latent_dims, cat_dims)
    x = gen.predict(latent_points, verbose=0)

    y = np.zeros((batch, 1))
    return x, y


def train_disc(gen, disc, cfg):
    batch = cfg["batch"]
    realX, realY = generate_real_samples(cfg["dataset"], batch // 2)
    fakeX, fakeY = generate_fake_samples(gen, cfg["latent_dims"], batch // 2)

    disc_loss_real, disc_acc_real = disc.train_on_batch(realX, realY)
    disc_loss_fake, disc_acc_fake = disc.train_on_batch(fakeX, fakeY)

    return disc_loss_real, disc_loss_fake, disc_acc_real, disc_acc_fake


def train_gen(gen, gan, cfg):
    batch = cfg["batch"]
    cat_dims = cfg["category_dims"]
    latent_dims = cfg["latent_dims"]

    latent_points, _, categories = generate_latent_points(batch, latent_dims, cat_dims)
    realY = np.ones((batch, 1))

    _, gen_loss, aux_loss = gan.train_on_batch(latent_points, [realY, categories])

    return gen_loss, aux_loss


def summarize_performance(gen, cfg, epoch):
    batch = 100
    cat_dims = cfg["category_dims"]
    latent_dims = cfg["latent_dims"]

    _, latent_points, _ = generate_latent_points(batch, latent_dims, cat_dims)
    categories = [x for x in range(int(np.sqrt(batch))) for _ in range(int(np.sqrt(batch)))]
    one_hot = tf.one_hot(categories, cat_dims)
    latent_and_cats = np.hstack([latent_points, one_hot])

    images = gen.predict(latent_and_cats)

    save_images(images, f"{epoch:03d}")


def train(gen, disc, aux, gan, cfg):
    writer = define_writer()
    ds = cfg["dataset"]
    batch = cfg["batch"]
    steps_per_epoch = ds.shape[0] // batch

    for epoch in range(cfg["epochs"]):
        list_disc_loss_real = list()
        list_disc_loss_fake = list()
        list_disc_acc_real = list()
        list_disc_acc_fake = list()
        list_gen_loss = list()
        list_aux_loss = list()

        for i in range(steps_per_epoch):
            disc_loss_real, disc_loss_fake, disc_acc_real, disc_acc_fake = train_disc(gen, disc, cfg)
            gen_loss, aux_loss = train_gen(gen, gan, cfg)

            list_disc_loss_real.append(disc_loss_real)
            list_disc_loss_fake.append(disc_loss_fake)
            list_disc_acc_real.append(disc_acc_real)
            list_disc_acc_fake.append(disc_acc_fake)
            list_gen_loss.append(gen_loss)
            list_aux_loss.append(aux_loss)

            print(f"{epoch} {i}/{steps_per_epoch}: d_loss {disc_loss_real:.3f}/{disc_loss_fake:.3f}, d_acc {disc_acc_real:.3f}/{disc_acc_fake:.3f}, gan_loss {gen_loss:.3f}/{aux_loss:.3f}")

        tf.summary.scalar("disc_loss_real", np.mean(list_disc_loss_real))
        tf.summary.scalar("disc_loss_fake", np.mean(list_disc_loss_fake))
        tf.summary.scalar("disc_acc_real", np.mean(list_disc_acc_real))
        tf.summary.scalar("disc_acc_fake", np.mean(list_disc_acc_fake))
        tf.summary.scalar("gen_loss", np.mean(list_gen_loss))
        tf.summary.scalar("aux_loss", np.mean(list_aux_loss))
        summarize_performance(gen, cfg, epoch)

def prep():
    if not os.path.isdir("progress"):
        os.mkdir("progress")


def main():
    prep()

    cfg = {
        "epochs": 20,
        "batch": 64,
        "latent_dims": 62,
        "category_dims": 10,
        "dataset": load_dataset(),
    }
    save_images(cfg["dataset"][:25], "init")

    disc, aux = define_discriminator(cfg["dataset"].shape[1:], cfg["category_dims"])
    gen = define_generator(cfg["latent_dims"], cfg["category_dims"])
    gan = define_gan(gen, disc, aux)
    # gan.summary(show_trainable=True, expand_nested=True)

    tf.keras.utils.plot_model(disc, "progress/disc.png", show_shapes=True)
    tf.keras.utils.plot_model(aux, "progress/aux_model.png", show_shapes=True)
    tf.keras.utils.plot_model(gen, "progress/gen.png", show_shapes=True)
    tf.keras.utils.plot_model(gan, "progress/gan.png", show_shapes=True)

    train(gen, disc, aux, gan, cfg)

    return


if __name__ == "__main__":
    main()
