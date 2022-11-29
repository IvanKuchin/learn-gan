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

    out_aux_cat = tf.keras.layers.Dense(cat_dims, activation="softmax")(act4)
    out_aux_cont1 = tf.keras.layers.Dense(1, activation="linear")(act4)
    out_aux_cont2 = tf.keras.layers.Dense(1, activation="linear")(act4)

    aux = tf.keras.Model(inputs=inputs, outputs=[out_aux_cat, out_aux_cont1, out_aux_cont2], name="aux")

    return disc, aux


def define_generator(latent_dims, cat_dims, num_cont_control):
    init = tf.keras.initializers.RandomNormal(stddev=0.02)

    inputs = tf.keras.Input((latent_dims + cat_dims + num_cont_control))
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
    aux_out1, aux_out2, aux_out3 = aux(gen_out)

    model = tf.keras.Model(inputs=inputs, outputs=[disc_out, aux_out1, aux_out2, aux_out3], name="gan")
    opt = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss=["binary_crossentropy", "categorical_crossentropy", "mse", "mse"], optimizer=opt)

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


def generate_latent_points(batch, latent_dims, cat_dims, num_cont_controls):
    latent_points = np.random.randn(batch * latent_dims)
    latent_points = np.reshape(latent_points, (batch, latent_dims))

    categories = tf.one_hot(np.random.randint(0, cat_dims, batch), cat_dims)

    cont_controls = np.random.rand(batch * num_cont_controls) * 2.0 - 1
    cont_controls = np.reshape(cont_controls, (batch, num_cont_controls))

    latent_and_cats = np.hstack([latent_points, categories.numpy(), cont_controls])
    return latent_and_cats, latent_points, categories, cont_controls


def generate_fake_samples(gen, latent_dims, num_cont_controls, batch):
    input_length = gen.input.shape[-1]
    cat_dims = input_length - latent_dims - num_cont_controls

    latent_points, _, _, _ = generate_latent_points(batch, latent_dims, cat_dims, num_cont_controls)
    x = gen.predict(latent_points, verbose=0)

    y = np.zeros((batch, 1))
    return x, y


def train_disc(gen, disc, cfg):
    batch = cfg["batch"]
    realX, realY = generate_real_samples(cfg["dataset"], batch // 2)
    fakeX, fakeY = generate_fake_samples(gen, cfg["latent_dims"], cfg["num_cont_controls"], batch // 2)

    disc_loss_real, disc_acc_real = disc.train_on_batch(realX, realY)
    disc_loss_fake, disc_acc_fake = disc.train_on_batch(fakeX, fakeY)

    return disc_loss_real, disc_loss_fake, disc_acc_real, disc_acc_fake


def train_gen(gen, gan, cfg):
    batch = cfg["batch"]
    cat_dims = cfg["category_dims"]
    latent_dims = cfg["latent_dims"]
    num_cont_controls = cfg["num_cont_controls"]

    latent_points, _, categories, cont_controls = generate_latent_points(batch, latent_dims, cat_dims, num_cont_controls)
    realY = np.ones((batch, 1))

    cont_control1 = np.expand_dims(cont_controls[:, 0], -1)
    cont_control2 = np.expand_dims(cont_controls[:, 1], -1)

    _, gen_loss, aux_loss1, aux_loss2, aux_loss3 = gan.train_on_batch(latent_points, [realY, categories, cont_control1, cont_control2])

    return gen_loss, aux_loss1, aux_loss2, aux_loss3


def summarize_performance(gen, cfg, epoch):
    batch = 100
    cat_dims = cfg["category_dims"]
    latent_dims = cfg["latent_dims"]
    num_cont_controls = cfg["num_cont_controls"]
    sqrt = int(np.sqrt(batch))

    _, latent_points, _, _ = generate_latent_points(batch, latent_dims, cat_dims, num_cont_controls)
    categories = [x for x in range(sqrt) for _ in range(sqrt)]
    one_hot = tf.one_hot(categories, cat_dims)

    ls = np.linspace(-1, 1, sqrt)
    ls = np.expand_dims(ls, 0)
    ls = np.repeat(ls, sqrt, axis=0)
    ls = np.reshape(ls, (1, -1))
    ls = np.repeat(ls, num_cont_controls, axis=0)
    cont_controls = ls.transpose()

    latent_and_cats = np.hstack([latent_points, one_hot, cont_controls])

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
        list_aux_loss1 = list()
        list_aux_loss2 = list()
        list_aux_loss3 = list()

        for i in range(steps_per_epoch):
            disc_loss_real, disc_loss_fake, disc_acc_real, disc_acc_fake = train_disc(gen, disc, cfg)
            gen_loss, aux_loss1, aux_loss2, aux_loss3 = train_gen(gen, gan, cfg)

            list_disc_loss_real.append(disc_loss_real)
            list_disc_loss_fake.append(disc_loss_fake)
            list_disc_acc_real.append(disc_acc_real)
            list_disc_acc_fake.append(disc_acc_fake)
            list_gen_loss.append(gen_loss)
            list_aux_loss1.append(aux_loss1)
            list_aux_loss2.append(aux_loss2)
            list_aux_loss3.append(aux_loss3)

            print(f"{epoch} {i}/{steps_per_epoch}: d_loss {disc_loss_real:.3f}/{disc_loss_fake:.3f}, d_acc {disc_acc_real:.3f}/{disc_acc_fake:.3f}, gan_loss {gen_loss:.3f}/{aux_loss1:.3f}/{aux_loss2:.3f}/{aux_loss3:.3f}")

        tf.summary.scalar("disc_loss_real", np.mean(list_disc_loss_real))
        tf.summary.scalar("disc_loss_fake", np.mean(list_disc_loss_fake))
        tf.summary.scalar("disc_acc_real", np.mean(list_disc_acc_real))
        tf.summary.scalar("disc_acc_fake", np.mean(list_disc_acc_fake))
        tf.summary.scalar("gen_loss", np.mean(list_gen_loss))
        tf.summary.scalar("aux_loss_categorical", np.mean(list_aux_loss1))
        tf.summary.scalar("aux_loss_continuous1", np.mean(list_aux_loss2))
        tf.summary.scalar("aux_loss_continuous2", np.mean(list_aux_loss3))
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
        "num_cont_controls": 2,
        "dataset": load_dataset(),
    }
    save_images(cfg["dataset"][:25], "init")

    disc, aux = define_discriminator(cfg["dataset"].shape[1:], cfg["category_dims"])
    gen = define_generator(cfg["latent_dims"], cfg["category_dims"], cfg["num_cont_controls"])
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
