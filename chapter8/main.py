import datetime
import io

import tensorflow as tf
import numpy as np
from matplotlib import pyplot

def define_discriminator():
    inputs = tf.keras.Input(shape=(32, 32, 3), name="input")
    conv1 = tf.keras.layers.Conv2D(32, (3, 3), strides=(2, 2), padding="same", name="conv1")(inputs)
    act1 = tf.keras.layers.LeakyReLU(alpha=0.2, name="act1")(conv1)
    conv2 = tf.keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same", name="conv2")(act1)
    act2 = tf.keras.layers.LeakyReLU(alpha=0.2, name="act2")(conv2)
    conv3 = tf.keras.layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same", name="conv3")(act2)
    act3 = tf.keras.layers.LeakyReLU(alpha=0.2, name="act3")(conv3)
    flat = tf.keras.layers.Flatten()(act3)
    drop1 = tf.keras.layers.Dropout(0.4, name="drop1")(flat)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="dense1")(drop1)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="disc")
    opt1 = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss="binary_crossentropy", optimizer=opt1, metrics=["accuracy"])

    return model


def define_generator(latent_dims):
    inputs = tf.keras.Input(shape=latent_dims)
    dense1 = tf.keras.layers.Dense(256 * 4 * 4, name="dense1")(inputs)
    act0 = tf.keras.layers.LeakyReLU(alpha=0.2, name="act0")(dense1)
    reshape = tf.keras.layers.Reshape((4, 4, 256), name="reshape1")(act0)
    conv1tr = tf.keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same", name="conv1tr")(reshape)
    act1 = tf.keras.layers.LeakyReLU(alpha=0.2, name="act1")(conv1tr)
    conv2tr = tf.keras.layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding="same", name="conv2tr")(act1)
    act2 = tf.keras.layers.LeakyReLU(alpha=0.2, name="act2")(conv2tr)
    conv3tr = tf.keras.layers.Conv2DTranspose(32, (4, 4), strides=(2, 2), padding="same", name="conv3tr")(act2)
    act3 = tf.keras.layers.LeakyReLU(alpha=0.2, name="act3")(conv3tr)
    outputs = tf.keras.layers.Conv2D(3, (3, 3), strides=(1, 1), padding="same", name="conv4", activation="tanh")(act3)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="generator")
    return model


def define_gan(disc, gen):
    disc.trainable = False
    inputs = tf.keras.Input(shape=gen.input.shape[1:])
    gen_out = gen(inputs)
    disc_out = disc(gen_out)

    model = tf.keras.Model(inputs=inputs, outputs=disc_out, name="gan")
    opt1 = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss="binary_crossentropy", optimizer=opt1, metrics=["accuracy"])

    return model


def generate_latent_points(batch, latent_dims):
    rnd = np.random.random(batch * latent_dims) * 2 - 1
    rnd = np.reshape(rnd, (batch, latent_dims))
    return rnd


def generate_fake_samples(gen, batch):
    latent_dims = gen.input.shape[1]
    points = generate_latent_points(batch, latent_dims)
    x = gen.predict(points, verbose=0)
    y = np.zeros((batch, 1))
    return x, y

def generate_real_samples(ds, batch):
    ix = np.random.randint(0, ds.shape[0], batch)
    x = ds[ix]
    y = np.ones((batch, 1))
    return x, y


def train_discriminator(disc, gen, ds, batch):
    half_batch = batch // 2
    realX, realY = generate_real_samples(ds=ds, batch=half_batch)
    fakeX, fakeY = generate_fake_samples(gen=gen, batch=half_batch)

    x = np.vstack((realX, fakeX))
    y = np.vstack((realY, fakeY))
    loss, _ = disc.train_on_batch(x, y)

    return loss


def train_gan(gan, batch):
    latent_dims = gan.input.shape[1]
    x = generate_latent_points(batch, latent_dims)
    y = np.ones((batch, 1))
    loss, _ = gan.train_on_batch(x, y)
    return loss


def train(gen, disc, gan, cfg):
    (trainX, _), (testX, _) = cfg["dataset"]

    steps_per_epoch = trainX.shape[0] // cfg["batch"]

    writer = create_writer(logdir="logs")

    for i in range(cfg["epochs"]):
        for j in range(steps_per_epoch):
        # for j in range(3):
            disc_loss = train_discriminator(disc=disc, gen=gen, ds=trainX, batch=cfg["batch"])
            gan_loss = train_gan(gan, batch=cfg["batch"])

            print(f"{i}, {j}/{steps_per_epoch}: {disc_loss=:.3f}, {gan_loss=:.3f}")

        if i % cfg["eval_freq"] == 0:
            (d_loss, g_loss, real_acc, fake_acc, image) = summarize_performance(disc=disc, gan=gan, gen=gen, cfg=cfg,
                                                                                step=i)
            with writer.as_default():
                tf.summary.scalar(name="disc_loss", data=d_loss, step=i)
                tf.summary.scalar(name="gan_loss", data=g_loss, step=i)
                tf.summary.scalar(name="real_acc", data=real_acc, step=i)
                tf.summary.scalar(name="fake_acc", data=fake_acc, step=i)
                tf.summary.image(name="image", data=image, step=i)


def create_writer(logdir):
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    dir = f"{logdir}/{ts}"
    writer = tf.summary.create_file_writer(dir)
    return writer


def summarize_performance(disc, gen, gan, cfg, step):
    latent_points = generate_latent_points(cfg["batch"], gen.input.shape[1])

    (trainX, _), (_, _) = cfg["dataset"]

    realX, realY = generate_real_samples(trainX, cfg["batch"])
    fakeX, fakeY = generate_fake_samples(gen, cfg["batch"])

    x = np.vstack((realX, fakeX))
    y = np.vstack((realY, fakeY))

    _, real_acc = disc.evaluate(realX, realY)
    _, fake_acc = disc.evaluate(fakeX, fakeY)

    d_loss, _ = disc.evaluate(x, y)
    g_loss, _ = gan.evaluate(latent_points, realY)

    image = save_images(fakeX[:16], str(step))

    return d_loss, g_loss, real_acc, fake_acc, image

def save_images(ds, name):
    # normalization
    ds_min = np.min(ds)
    ds_max = np.max(ds)
    pics = (ds - ds_min) / (ds_max - ds_min)

    pic_square = int(np.sqrt(ds.shape[0]))

    for i in range(pic_square**2):
        pyplot.subplot(pic_square, pic_square, i + 1)
        pyplot.axis(False)
        pyplot.imshow(pics[i])

    pyplot.savefig(f"progress/{name}.png")

    buf = io.BytesIO()
    pyplot.savefig(buf, format="png")
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=3)
    image = np.expand_dims(image, 0)

    pyplot.close()

    return image

if __name__ == '__main__':
    config = {
        'latent_dims': 100,
        'epochs': 300,
        'batch': 256,
        'eval_freq': 1,
    }

    dataset = tf.keras.datasets.cifar10.load_data()
    (trainX, _), (testX, _) = dataset
    trainX = trainX / 255 * 2 - 1
    testX = testX / 255 * 2 - 1
    config["dataset"] = (trainX, _), (testX, _)

    d_model = define_discriminator()
    g_model = define_generator(config["latent_dims"])
    gan_model = define_gan(d_model, g_model)
    # gan_model.summary(expand_nested=True, show_trainable=True)
    train(gen=g_model, disc=d_model, gan=gan_model, cfg=config)
