import datetime
import os.path
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from matplotlib import pyplot


def load_dataset():
    (horses, zebras) = np.load("dataset.npy", allow_pickle=True)
    horses = horses / 255.0 * 2.0 - 1
    zebras = zebras / 255.0 * 2.0 - 1
    return (horses, zebras)


def save_images(images, name):
    cardinality = len(images.shape)
    if cardinality != 4:
        print(f"images cardinality must be 4, but given shape is {images.shape}")
        return

    edge = int(np.sqrt(images.shape[0]))
    min, max = images.min(), images.max()
    images = (images - min) / (max - min)
    pyplot.figure(figsize=(20, 20), dpi=240)
    for i in range(edge**2):
        pyplot.subplot(edge, edge, i+1)
        pyplot.axis(False)
        pyplot.imshow(images[i])
    pyplot.savefig(f"progress/{name}.png")
    pyplot.close()


def prep():
    if not os.path.isdir("progress"):
        os.mkdir("progress")


def define_discriminator(image_shape, name):
    init = tf.keras.initializers.RandomNormal(stddev=0.02)
    inputs = tf.keras.Input(image_shape, name="image")

    # 128x128
    conv1 = tf.keras.layers.Conv2D(64, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init)(inputs)
    norm1 = tfa.layers.InstanceNormalization(axis=-1)(conv1)
    act1 = tf.keras.layers.LeakyReLU(alpha=0.2)(norm1)

    # 64x64
    conv2 = tf.keras.layers.Conv2D(128, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init)(act1)
    norm2 = tfa.layers.InstanceNormalization(axis=-1)(conv2)
    act2 = tf.keras.layers.LeakyReLU(alpha=0.2)(norm2)

    # 32x32
    conv3 = tf.keras.layers.Conv2D(256, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init)(act2)
    norm3 = tfa.layers.InstanceNormalization(axis=-1)(conv3)
    act3 = tf.keras.layers.LeakyReLU(alpha=0.2)(norm3)

    # 16x16
    conv4 = tf.keras.layers.Conv2D(512, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init)(act3)
    norm4 = tfa.layers.InstanceNormalization(axis=-1)(conv4)
    act4 = tf.keras.layers.LeakyReLU(alpha=0.2)(norm4)

    # 16x16
    conv5 = tf.keras.layers.Conv2D(512, (4, 4), strides=(1, 1), padding="same", kernel_initializer=init)(act4)
    norm5 = tfa.layers.InstanceNormalization(axis=-1)(conv5)
    act5 = tf.keras.layers.LeakyReLU(alpha=0.2)(norm5)

    # 16x16
    patch_out = tf.keras.layers.Conv2D(1, (4, 4), strides=(1, 1), padding="same", kernel_initializer=init)(act5)

    model = tf.keras.Model(inputs=inputs, outputs=patch_out, name=f"discriminator_{name}")
    opt = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss="mse", optimizer=opt, loss_weights=[0.5])
    return model


def residual_block(filters, inputs):
    init = tf.keras.initializers.RandomNormal(stddev=0.02)

    conv1 = tf.keras.layers.Conv2D(filters, (3, 3), padding="same", kernel_initializer=init)(inputs)
    norm1 = tfa.layers.InstanceNormalization(axis=-1)(conv1)
    act1 = tf.keras.layers.LeakyReLU(alpha=0.2)(norm1)

    conv2 = tf.keras.layers.Conv2D(filters, (3, 3), padding="same", kernel_initializer=init)(act1)
    norm2 = tfa.layers.InstanceNormalization(axis=-1)(conv2)

    concat = tf.keras.layers.Concatenate()([norm2, inputs])

    return concat


def define_generator(image_shape, name):
    init = tf.keras.initializers.RandomNormal(stddev=0.02)
    inputs = tf.keras.Input(image_shape, name="image")

    conv1 = tf.keras.layers.Conv2D(64, (7, 7), strides=(1, 1), padding="same", kernel_initializer=init)(inputs)
    norm1 = tfa.layers.InstanceNormalization(axis=-1)(conv1)
    act1 = tf.keras.layers.LeakyReLU(alpha=0.2)(norm1)

    conv2 = tf.keras.layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same", kernel_initializer=init)(act1)
    norm2 = tfa.layers.InstanceNormalization(axis=-1)(conv2)
    act2 = tf.keras.layers.LeakyReLU(alpha=0.2)(norm2)

    conv3 = tf.keras.layers.Conv2D(256, (3, 3), strides=(2, 2), padding="same", kernel_initializer=init)(act2)
    norm3 = tfa.layers.InstanceNormalization(axis=-1)(conv3)
    act3 = tf.keras.layers.LeakyReLU(alpha=0.2)(norm3)

    res = act3
    for _ in range(6):
        res = residual_block(256, res)

    conv4tr = tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding="same", kernel_initializer=init)(res)
    norm4 = tfa.layers.InstanceNormalization(axis=-1)(conv4tr)
    act4 = tf.keras.layers.LeakyReLU(alpha=0.2)(norm4)

    conv5tr = tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same", kernel_initializer=init)(act4)
    norm5 = tfa.layers.InstanceNormalization(axis=-1)(conv5tr)
    act5 = tf.keras.layers.LeakyReLU(alpha=0.2)(norm5)

    conv6 = tf.keras.layers.Conv2D(3, (7, 7), padding="same", kernel_initializer=init)(act5)
    norm6 = tfa.layers.InstanceNormalization(axis=-1)(conv6)
    act6 = tf.keras.activations.tanh(norm6)

    model = tf.keras.Model(inputs=inputs, outputs=act6, name=f"generator_{name}")

    return model


def define_gan(gen_AB, disc_B, gen_BA):
    gen_AB.trainable = True
    disc_B.trainable = False
    gen_BA.trainable = False

    image_shape = gen_AB.input.shape[1:]
    XrealA = tf.keras.Input(image_shape, name="realA")
    XrealB = tf.keras.Input(image_shape, name="realB")
    XfakeB = gen_AB(XrealA)
    XfakeA = gen_BA(XrealB)
    fakeness = disc_B(XfakeB)
    identB = gen_AB(XrealB)
    outputF = gen_BA(XfakeB)
    outputB = gen_AB(XfakeA)

    model = tf.keras.Model(inputs=[XrealA, XrealB], outputs=[fakeness, identB, outputF, outputB], name="gan")
    opt = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss=["mse", "mae", "mae", "mae"], optimizer=opt, loss_weights=[1, 5, 10, 10])
    return model


def define_writer():
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return tf.summary.create_file_writer(f"logs/{ts}")


def generate_real_samples(ds, batch):
    ix = np.random.randint(0, ds.shape[0], batch)
    return ds[ix]


def generate_fake_samples(images, gen):
    return gen.predict(images, verbose=0)


def generate_yreal(disc_A, batch):
    out_shape = disc_A.output.shape[1:]
    return np.ones((batch,) + out_shape)


def generate_yfake(disc_A, batch):
    out_shape = disc_A.output.shape[1:]
    return np.zeros((batch,) + out_shape)


def summarize_performance(gen_AB, gen_BA, dsA, dsB, epoch):
    batch = 25
    ix = np.random.randint(0, dsA.shape[0], batch)
    realA = dsA[ix]
    ix = np.random.randint(0, dsB.shape[0], batch)
    realB = dsB[ix]

    fakeB = generate_fake_samples(realA, gen_AB)
    fakeA = generate_fake_samples(realB, gen_BA)

    save_images(fakeA, f"a_{epoch}")
    save_images(fakeB, f"b_{epoch}")
    gen_AB.save(f"progress/ab_{epoch}.h5")
    gen_BA.save(f"progress/ba_{epoch}.h5")
    pass


def train(gen_AB, gen_BA, disc_A, disc_B, gan_AB, gan_BA, cfg):
    writer = define_writer()

    batch = cfg["batch"]
    ds_size = np.max([cfg["dataset"][0].shape[0], cfg["dataset"][1].shape[0]])
    steps_per_epoch = ds_size // batch

    Yreal = generate_yreal(disc_A, batch)
    Yfake = generate_yfake(disc_A, batch)

    for epoch in range(cfg["epochs"]):
        list_loss_discA_real = list()
        list_loss_discA_fake = list()
        list_loss_discB_real = list()
        list_loss_discB_fake = list()
        list_loss_genAB = list()
        list_loss_genBA = list()
        for step in range(steps_per_epoch):
            XrealA = generate_real_samples(cfg["dataset"][0], batch)
            XrealB = generate_real_samples(cfg["dataset"][1], batch)

            XfakeA = generate_fake_samples(XrealB, gen_BA)
            XfakeB = generate_fake_samples(XrealA, gen_AB)

            loss_discA_real = disc_A.train_on_batch(XrealA, Yreal)
            loss_discA_fake = disc_A.train_on_batch(XfakeA, Yfake)
            loss_discB_real = disc_B.train_on_batch(XrealB, Yreal)
            loss_discB_fake = disc_B.train_on_batch(XfakeB, Yfake)

            loss_genAB, _, _, _, _ = gan_AB.train_on_batch([XrealA, XrealB], [Yreal, XrealB, XrealA, XrealB])
            loss_genBA, _, _, _, _ = gan_BA.train_on_batch([XrealB, XrealA], [Yreal, XrealA, XrealB, XrealA])

            print(f"{epoch} {step:3d}/{steps_per_epoch}: dA {loss_discA_real:.3f}/{loss_discA_fake:.3f} dB {loss_discB_real:.3f}/{loss_discB_fake:.3f} gan {loss_genAB:.3f}/{loss_genBA:.3f}")

            list_loss_discA_real.append(loss_discA_real)
            list_loss_discA_fake.append(loss_discA_fake)
            list_loss_discB_real.append(loss_discB_real)
            list_loss_discB_fake.append(loss_discB_fake)
            list_loss_genAB.append(loss_genAB)
            list_loss_genBA.append(loss_genBA)

        summarize_performance(gen_AB, gen_BA, cfg["dataset"][0], cfg["dataset"][1], f"{epoch:03d}")
        with writer.as_default():
            tf.summary.scalar("loss_discA_real", np.mean(list_loss_discA_real), step=epoch)
            tf.summary.scalar("loss_discA_fake", np.mean(list_loss_discA_fake), step=epoch)
            tf.summary.scalar("loss_discB_real", np.mean(list_loss_discB_real), step=epoch)
            tf.summary.scalar("loss_discB_fake", np.mean(list_loss_discB_fake), step=epoch)
            tf.summary.scalar("loss_genAB", np.mean(list_loss_genAB), step=epoch)
            tf.summary.scalar("loss_genBA", np.mean(list_loss_genBA), step=epoch)
    pass


def main():
    prep()
    cfg = {
        "batch": 8,
        "epochs": 100,
        "dataset": load_dataset()
    }

    image_shape = cfg["dataset"][0].shape[1:]
    disc_A = define_discriminator(image_shape, "A")
    disc_B = define_discriminator(image_shape, "B")
    gen_AB = define_generator(image_shape, "AB")
    gen_BA = define_generator(image_shape, "BA")
    gan_AB = define_gan(gen_AB, disc_B, gen_BA)
    gan_BA = define_gan(gen_BA, disc_A, gen_AB)

    # tf.keras.utils.plot_model(disc_A, f"progress/disc_A.png", show_shapes=True)
    # tf.keras.utils.plot_model(gen_AB, "progress/generator.png", show_shapes=True)
    # tf.keras.utils.plot_model(gan_AB, "progress/gan.png", show_shapes=True)

    train(gen_AB, gen_BA, disc_A, disc_B, gan_AB, gan_BA, cfg)

if __name__ == "__main__":
    main()
