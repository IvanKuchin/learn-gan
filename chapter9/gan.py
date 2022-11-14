import datetime
import io

import numpy as np
from matplotlib import pyplot
import tensorflow as tf

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

def define_discriminator_book(in_shape=(80, 80, 3)):
    model = Sequential()
    # normal
    model.add(Conv2D(128, (5, 5), padding='same', input_shape=in_shape))
    model.add(LeakyReLU(alpha=0.2))
    # downsample to 40x40
    model.add(Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # downsample to 20x20
    model.add(Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # downsample to 10x10
    model.add(Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # downsample to 5x5
    model.add(Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # classifier
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


# define the standalone generator model
def define_generator_book(latent_dim):
    model = Sequential()
    # foundation for 5x5 feature maps
    n_nodes = 128 * 5 * 5
    model.add(Dense(n_nodes, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((5, 5, 128)))
    # upsample to 10x10
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # upsample to 20x20
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # upsample to 40x40
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # upsample to 80x80
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # output layer 80x80x3
    model.add(Conv2D(3, (5, 5), activation='tanh', padding='same'))
    return model



def define_discriminator(shape):
    inputs = tf.keras.Input(shape=shape)
    conv0 = tf.keras.layers.Conv2D(32, (5, 5), strides=(1, 1), padding="same")(inputs) # 80x80
    act0 = tf.keras.layers.LeakyReLU(alpha=0.2)(conv0)
    conv1 = tf.keras.layers.Conv2D(32, (5, 5), strides=(2, 2), padding="same")(act0) # 40x40
    act1 = tf.keras.layers.LeakyReLU(alpha=0.2)(conv1)
    conv2 = tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding="same")(act1) # 20x20
    act2 = tf.keras.layers.LeakyReLU(alpha=0.2)(conv2)
    conv3 = tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding="same")(act2) # 10x10
    act3 = tf.keras.layers.LeakyReLU(alpha=0.2)(conv3)
    conv4 = tf.keras.layers.Conv2D(256, (5, 5), strides=(2, 2), padding="same")(act3) # 5x5
    act4 = tf.keras.layers.LeakyReLU(alpha=0.2)(conv4)
    flatten = tf.keras.layers.Flatten()(act4)
    drop1 = tf.keras.layers.Dropout(0.4)(flatten)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(drop1)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="discriminator")
    opt1 = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss="binary_crossentropy", optimizer=opt1, metrics=["accuracy"])

    return model

def define_generator(latent_dims):
    inputs = tf.keras.Input(shape=latent_dims)
    dense1 = tf.keras.layers.Dense(256 * 5 * 5)(inputs)
    act_d1 = tf.keras.layers.LeakyReLU(alpha=0.2)(dense1)
    reshape = tf.keras.layers.Reshape((5, 5, 256))(act_d1)
    conv1tr = tf.keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same")(reshape)
    act1 = tf.keras.layers.LeakyReLU(alpha=0.2)(conv1tr)
    conv2tr = tf.keras.layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding="same")(act1)
    act2 = tf.keras.layers.LeakyReLU(alpha=0.2)(conv2tr)
    conv3tr = tf.keras.layers.Conv2DTranspose(32, (4, 4), strides=(2, 2), padding="same")(act2)
    act3 = tf.keras.layers.LeakyReLU(alpha=0.2)(conv3tr)
    conv4tr = tf.keras.layers.Conv2DTranspose(16, (4, 4), strides=(2, 2), padding="same")(act3)
    act4 = tf.keras.layers.LeakyReLU(alpha=0.2)(conv4tr)
    conv5 = tf.keras.layers.Conv2D(3, (5, 5), strides=(1, 1), padding="same", activation="tanh")(act4)

    model = tf.keras.Model(inputs=inputs, outputs=conv5, name="generator")
    return model


def define_gan(gen, disc):
    disc.trainable = False

    inputs = tf.keras.Input(shape=gen.input.shape[1:])
    gen_out = gen(inputs)
    disc_out = disc(gen_out)

    model = tf.keras.Model(inputs=inputs, outputs=disc_out)

    # model = tf.keras.models.Sequential()
    # model.add(gen)
    # model.add(disc)

    opt1 = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss="binary_crossentropy", optimizer=opt1, metrics=["accuracy"])

    return model


def save_images(ds, name):
    cardinality = len(ds.shape)

    if cardinality != 4:
        print(f"dataset cardinality must be 4, but given {ds.shape}")
        return

    ds_min = ds.min()
    ds_max = ds.max()
    ds = (ds - ds_min) / (ds_max - ds_min)

    square = int(np.sqrt(ds.shape[0]))

    for i in range(square**2):
        pyplot.subplot(square, square, i + 1)
        pyplot.axis(False)
        pyplot.imshow(ds[i])
    pyplot.savefig(f"progress/{name}.png")

    # convert to tensorboard-format
    buf = io.BytesIO()
    pyplot.savefig(buf, format="png")
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=3)
    image = np.expand_dims(image, 0)

    pyplot.close()

    return image


def generate_latent_points(batch, latent_dims):
    r = np.random.random(batch * latent_dims) * 2 - 1
    points = np.reshape(r, (batch, latent_dims))
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
    realX, realY = generate_real_samples(cfg["ds"], cfg["batch"] // 2)
    fakeX, fakeY = generate_fake_samples(gen, cfg["batch"] // 2)
    # print(f"disc: real {realX.min():.3f}/{realX.mean():.3f}/{realX.max():.3f}, fake: {fakeX.min():.3f}/{fakeX.mean():.3f}/{fakeX.max():.3f}")
    loss_real, _ = disc.train_on_batch(realX, realY)
    loss_fake, _ = disc.train_on_batch(fakeX, fakeY)
    return loss_real, loss_fake
    # x = np.vstack((realX, fakeX))
    # y = np.vstack((realY, fakeY))
    # loss, _ = disc.train_on_batch(x, y)
    # return loss


def train_gan(gan, cfg):
    x = generate_latent_points(cfg["batch"], cfg["latent_dims"])
    y = np.ones((cfg["batch"], 1))
    # print(f"gan : {x.min():.3f}/{x.mean():.3f}/{x.max():.3f}")
    loss, _ = gan.train_on_batch(x, y)
    return loss


def summarize_performance(disc, gen, gan, cfg, step):
    realX, realY = generate_real_samples(cfg["ds"], cfg["batch"])
    fakeX, fakeY = generate_fake_samples(gen, cfg["batch"])

    disc_loss_real, disc_acc_real = disc.evaluate(realX, realY)
    disc_loss_fake, disc_acc_fake = disc.evaluate(fakeX, fakeY)

    x = np.vstack((realX, fakeX))
    y = np.vstack((realY, fakeY))

    latent_points = generate_latent_points(cfg["batch"], cfg["latent_dims"])

    disc_loss, _ = disc.evaluate(x, y)
    gan_loss, _ = gan.evaluate(latent_points, realY)

    image = save_images(fakeX[:16], f"{step:03d}")

    return disc_loss, disc_loss_real, disc_loss_fake, gan_loss, disc_acc_real, disc_acc_fake, image

def build_writer(logdir):
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = tf.summary.create_file_writer(f"{logdir}/{ts}")
    return writer

def train(gen, disc, gan, cfg):
    writer = build_writer("logs")
    steps_per_epoch = cfg["ds"].shape[0] // cfg["batch"]
    for epoch in range(cfg["epochs"]):
        for step in range(steps_per_epoch):
            # for step in range(3):
            d_loss_real, d_loss_fake = train_disc(gen, disc, cfg)
            gan_loss = train_gan(gan, cfg)
            print(f"{epoch}, {step}/{steps_per_epoch}: {d_loss_real=:.3f} {d_loss_fake=:.3f} {gan_loss=:.3f}")

        disc_loss, disc_loss_real, disc_loss_fake, gan_loss, disc_acc_real, disc_acc_fake, image = summarize_performance(disc, gen, gan, cfg, epoch)
        with writer.as_default():
            tf.summary.scalar("disc_loss", disc_loss, step=epoch)
            tf.summary.scalar("disc_loss_real", disc_loss_real, step=epoch)
            tf.summary.scalar("disc_loss_fake", disc_loss_fake, step=epoch)
            tf.summary.scalar("gan_loss", gan_loss, step=epoch)
            tf.summary.scalar("disc_acc_real", disc_acc_real, step=epoch)
            tf.summary.scalar("disc_acc_fake", disc_acc_fake, step=epoch)
            tf.summary.image("generator", image, step=epoch)

        if (epoch + 1) % 10 == 0:
            gen.save(f"models/gen_{epoch+1:03d}.h5")
    return


def main():
    cfg = {
        "epochs": 100,
        "batch": 128,
        "latent_dims": 100,
        "dataset_file": "dataset/celeba.npy",
    }

    # load dataset
    ds = np.load(cfg["dataset_file"])
    ds_cardinality = len(ds.shape)
    if ds_cardinality != 4:
        print("dataset cardinality must be 4")
        return

    # convert to float
    ds = (ds - ds.min()) / (ds.max() - ds.min()) * 2 - 1
    cfg["ds"] = ds

    d = define_discriminator(ds.shape[1:])
    g = define_generator(cfg["latent_dims"])
    gan = define_gan(g, d)
    # gan.summary(show_trainable=True)

    train(gen=g, disc=d, gan=gan, cfg=cfg)


if __name__ == "__main__":
    main()
