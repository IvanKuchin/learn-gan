import datetime
import os.path

import tensorflow as tf
import numpy as np
from matplotlib import pyplot


def save_images(images, fname):
    cardinality = len(images.shape)
    if cardinality != 4:
        print(f"cardinality must be 4, but given shape is {images.shape}")
        return
    min, max = images.min(), images.max()
    images = (images - min) / (max - min)
    edge = int(np.sqrt(images.shape[0]))
    for i in range(edge**2):
        pyplot.subplot(edge, edge, i+1)
        pyplot.axis(False)
        pyplot.imshow(images[i])
    pyplot.savefig(f"progress/{fname}.png")
    pyplot.close()


def prep(dir_name):
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)


def load_ds(fname):
    ds = np.load(fname)
    ds = ds / 255.0 * 2.0 - 1
    return ds


def define_writer(dir):
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = tf.summary.create_file_writer(f"{dir}/{ts}")
    return writer


def define_discriminator(image_shape):
    init = tf.keras.initializers.RandomNormal(stddev=0.02)

    input_sat = tf.keras.Input(image_shape, name="sat image")
    input_map = tf.keras.Input(image_shape, name="map image")

    concat = tf.keras.layers.Concatenate()([input_sat, input_map])

    # 128x128
    conv1 = tf.keras.layers.Conv2D(64, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init)(concat)
    act1 = tf.keras.layers.LeakyReLU(alpha=0.2)(conv1)

    # 64x64
    conv2 = tf.keras.layers.Conv2D(128, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init)(act1)
    bn2 = tf.keras.layers.BatchNormalization()(conv2)
    act2 = tf.keras.layers.LeakyReLU(alpha=0.2)(bn2)

    # 32x32
    conv3 = tf.keras.layers.Conv2D(256, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init)(act2)
    bn3 = tf.keras.layers.BatchNormalization()(conv3)
    act3 = tf.keras.layers.LeakyReLU(alpha=0.2)(bn3)

    # 16x16
    conv4 = tf.keras.layers.Conv2D(512, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init)(act3)
    bn4 = tf.keras.layers.BatchNormalization()(conv4)
    act4 = tf.keras.layers.LeakyReLU(alpha=0.2)(bn4)

    # 16x16
    conv5 = tf.keras.layers.Conv2D(512, (4, 4), strides=(1, 1), padding="same", kernel_initializer=init)(act4)
    bn5 = tf.keras.layers.BatchNormalization()(conv5)
    act5 = tf.keras.layers.LeakyReLU(alpha=0.2)(bn5)

    # 16x16
    conv6 = tf.keras.layers.Conv2D(1, (4, 4), strides=(1,1), padding="same", kernel_initializer=init)(act5)
    act6 = tf.keras.activations.sigmoid(conv6)

    model = tf.keras.Model(inputs=[input_sat, input_map], outputs=act6, name="discriminator")
    opt = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"], loss_weights=0.5)
    return model


def downsample_block(layer_in, n_filters, bn=True):
    init = tf.keras.initializers.RandomNormal(stddev=0.02)
    block = tf.keras.layers.Conv2D(n_filters, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init)(layer_in)
    if bn:
        block = tf.keras.layers.BatchNormalization()(block)
    block = tf.keras.layers.LeakyReLU(alpha=0.2)(block)
    return block


def upsample_block(layer_in, skip_in, n_filters, dropout=True):
    init = tf.keras.initializers.RandomNormal(stddev=0.2)
    conv = tf.keras.layers.Conv2DTranspose(n_filters, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init)(layer_in)
    bn = tf.keras.layers.BatchNormalization()(conv)
    if dropout:
        bn = tf.keras.layers.Dropout(0.5)(bn)
    concat = tf.keras.layers.Concatenate()([bn, skip_in])
    act = tf.keras.layers.ReLU()(concat)
    return act


def define_generator(image_shape):
    init = tf.keras.initializers.RandomNormal(stddev=0.02)

    image = tf.keras.Input(image_shape, name="image")
    down128 = downsample_block(image, 64, bn=False)
    down64 = downsample_block(down128, 128)
    down32 = downsample_block(down64, 256)
    down16 = downsample_block(down32, 512)
    down8 = downsample_block(down16, 512)
    down4 = downsample_block(down8, 512)
    down2 = downsample_block(down4, 512)

    # bottleneck
    bottleneck = tf.keras.layers.Conv2D(512, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init, activation="relu", name="bottleneck")(down2)

    up2   = upsample_block(bottleneck, down2, 512)
    up4   = upsample_block( up2,   down4, 512)
    up8   = upsample_block( up4,   down8, 512)
    up16  = upsample_block( up8,  down16, 512, dropout=False)
    up32  = upsample_block(up16,  down32, 256, dropout=False)
    up64  = upsample_block(up32,  down64, 128, dropout=False)
    up128 = upsample_block(up64, down128,  64, dropout=False)

    up256 = tf.keras.layers.Conv2DTranspose(3, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init, activation="tanh")(up128)

    model = tf.keras.Model(inputs=image, outputs=up256, name="generator")
    return model


def define_gan(gen, disc):
    for layer in disc.layers:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    image_shape = gen.input.shape[1:]
    image_real = tf.keras.Input(image_shape, name="real_image")
    image_fake = gen(image_real)
    fakeness = disc([image_real, image_fake])
    gan = tf.keras.Model(inputs=image_real, outputs=[fakeness, image_fake], name="gan")
    opt = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    gan.compile(loss=["binary_crossentropy", "mae"], optimizer=opt, loss_weights=[1, 100])
    return gan


def generate_real_samples(dataset, batch):
    number_of_samples = dataset[0].shape[0]
    ix = np.random.randint(0, number_of_samples, batch)
    return dataset[0][ix], dataset[1][ix]


def generate_fake_images(gen, sat_images):
    fake_images = gen.predict(sat_images, verbose=0)
    return fake_images


def summarize_performance(gen, ds, batch, name):
    sat_images, map_images = generate_real_samples(ds, batch)
    gan_images = gen.predict(sat_images, verbose=0)
    save_images(gan_images, f"gen_{name}")
    save_images(sat_images, f"sat_{name}")
    save_images(map_images, f"map_{name}")
    pass


def train(gen, disc, gan, cfg):
    writer = define_writer("logs")
    batch = cfg["batch"]
    steps_per_epoch = cfg["dataset"][0].shape[0] // batch

    disc_out_shape = (batch,) + disc.output.shape[1:]
    real_y = np.ones(disc_out_shape)
    fake_y = np.zeros(disc_out_shape)

    for epoch in range(cfg["epochs"]):
        list_loss_disc_real = list()
        list_acc_disc_real = list()
        list_loss_disc_fake = list()
        list_acc_disc_fake = list()
        list_logg_gan = list()
        for i in range(steps_per_epoch):
            sat_images, map_images = generate_real_samples(cfg["dataset"], batch)
            fake_images = generate_fake_images(gen, sat_images)
            loss_disc_real, acc_disc_real = disc.train_on_batch([sat_images, map_images], real_y)
            loss_disc_fake, acc_disc_fake = disc.train_on_batch([sat_images, fake_images], fake_y)

            logg_gan, loss_fakeness, loss_mae = gan.train_on_batch(sat_images, [real_y, map_images])

            print(f"{epoch} {i:4d}/{steps_per_epoch}: disc loss real/fake {loss_disc_real:.3f}/{loss_disc_fake:.3f} disc acc real/fake {acc_disc_real:.3f}/{acc_disc_fake:.3f} loss gan {logg_gan:.3f}/{loss_fakeness:.3f}/{loss_mae:.3f}")

            list_loss_disc_real.append(loss_disc_real)
            list_acc_disc_real.append(acc_disc_real)
            list_loss_disc_fake.append(loss_disc_fake)
            list_acc_disc_fake.append(acc_disc_fake)
            list_logg_gan.append(logg_gan)

        with writer.as_default():
            tf.summary.scalar("loss_disc_real", np.mean(list_loss_disc_real), step=epoch)
            tf.summary.scalar("acc_disc_real", np.mean(list_acc_disc_real), step=epoch)
            tf.summary.scalar("loss_disc_fake", np.mean(list_loss_disc_fake), step=epoch)
            tf.summary.scalar("acc_disc_fake", np.mean(list_acc_disc_fake), step=epoch)
            tf.summary.scalar("logg_gan", np.mean(list_logg_gan), step=epoch)

        summarize_performance(gen, cfg["dataset"], 100, f"{epoch:03d}")


def main():
    prep("progress")
    cfg = {
        "epochs": 100,
        "batch": 1,
        "dataset": load_ds("train.npy")
    }
    print(f"dataset shape is {cfg['dataset'].shape}")
    # save_images(cfg["dataset"][0][:25], "init_sat")
    # save_images(cfg["dataset"][1][:25], "init_map")
    disc = define_discriminator(cfg["dataset"][0].shape[1:])
    gen = define_generator(cfg["dataset"][0].shape[1:])
    gan = define_gan(gen, disc)
    # tf.keras.utils.plot_model(disc, "progress/disc.png", show_shapes=True)
    # tf.keras.utils.plot_model(gen, "progress/gen.png", show_shapes=True)
    # tf.keras.utils.plot_model(gan, "progress/gan.png", show_shapes=True)

    train(gen, disc, gan, cfg)


if __name__ == "__main__":
    main()
