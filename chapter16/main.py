import datetime
import os.path

import numpy as np
import tensorflow as tf
from matplotlib import pyplot


class ClipConstraint(tf.keras.constraints.Constraint):
    def __init__(self, clip_value):
        self.clip_value = clip_value

    def __call__(self, weights):
        return tf.clip_by_value(weights, -self.clip_value, self.clip_value)

    def get_config(self):
        return {"clip_value": self.clip_value}


def load_dataset():
    (trainX, trainY), (_, _) = tf.keras.datasets.mnist.load_data()
    ds = trainX[trainY == 7]
    ds = ds / 255.0 * 2.0 - 1
    ds = np.expand_dims(ds, -1)
    return ds


def loss(y_true, y_pred):
    l = y_true * y_pred
    return tf.reduce_mean(l)


def define_critic(in_shape):
    init = tf.keras.initializers.RandomNormal(stddev=0.02)
    clip = ClipConstraint(0.01)

    inputs = tf.keras.Input(shape=in_shape)

    conv1 = tf.keras.layers.Conv2D(64, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init, kernel_constraint=clip)(inputs)
    bn1 = tf.keras.layers.BatchNormalization()(conv1)
    act1 = tf.keras.layers.LeakyReLU(alpha=0.2)(bn1)

    conv2 = tf.keras.layers.Conv2D(128, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init, kernel_constraint=clip)(act1)
    bn2 = tf.keras.layers.BatchNormalization()(conv2)
    act2 = tf.keras.layers.LeakyReLU(alpha=0.2)(bn2)

    flat = tf.keras.layers.Flatten()(act2)
    dense1 = tf.keras.layers.Dense(1)(flat)

    model = tf.keras.Model(inputs=inputs, outputs=dense1, name="critic")
    opt = tf.keras.optimizers.RMSprop(learning_rate=0.00005)
    model.compile(loss=loss, optimizer=opt)

    return model


def generate_latent_points(batch, latent_dims):
    r = np.random.randn(batch * latent_dims)
    r = np.reshape(r, (batch, latent_dims))
    return r


def define_generator(latent_dims):
    init = tf.keras.initializers.RandomNormal(stddev=0.02)

    inputs = tf.keras.Input(latent_dims)

    dense0 = tf.keras.layers.Dense(128 * 7 * 7)(inputs)
    act0 = tf.keras.layers.LeakyReLU(alpha=0.2)(dense0)
    reshape0 = tf.keras.layers.Reshape((7, 7, 128))(act0)

    conv1tr = tf.keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init)(reshape0)
    bn1 = tf.keras.layers.BatchNormalization()(conv1tr)
    act1 = tf.keras.layers.LeakyReLU(alpha=0.2)(bn1)

    conv2tr = tf.keras.layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init)(act1)
    bn2 = tf.keras.layers.BatchNormalization()(conv2tr)
    act2 = tf.keras.layers.LeakyReLU(alpha=0.2)(bn2)

    conv3 = tf.keras.layers.Conv2D(1, (1, 1), strides=(1, 1), padding="same", kernel_initializer=init, activation="tanh")(act2)

    model = tf.keras.Model(inputs=inputs, outputs=conv3, name="generator")

    return model


def save_image(images, name):
    cardinality = len(images.shape)
    if cardinality != 4:
        print(f"dataset cardinality must be 4, but provided {images.shape}")
        return

    image_dir = "progress"
    if not os.path.isdir(image_dir):
        os.mkdir(image_dir)

    side = int(np.sqrt(images.shape[0]))
    for i in range(side ** 2):
        pyplot.subplot(side, side, i + 1)
        pyplot.axis(False)
        pyplot.imshow(images[i], cmap="gray_r")
    pyplot.savefig(f"{image_dir}/{name}.png")
    pyplot.close()


def define_writer():
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = f"logs/{ts}"
    writer = tf.summary.create_file_writer(logdir)
    return writer


def define_gan(gen, critic):
    for layer in critic.layers:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    inputs = tf.keras.Input(gen.input.shape[1:])
    gen_out = gen(inputs)
    critic_out = critic(gen_out)

    model = tf.keras.Model(inputs=inputs, outputs=critic_out, name="gan")
    opt = tf.keras.optimizers.RMSprop(learning_rate=0.00005)
    model.compile(loss=loss, optimizer=opt)
    return model


def generate_real_y(batch):
    return -np.ones((batch, 1))


def generate_fake_y(batch):
    return -generate_real_y(batch)


def generate_real_samples(ds, batch):
    ix = np.random.randint(0, ds.shape[0], batch)
    x = ds[ix]
    y = generate_real_y(batch)
    return x, y


def generate_fake_samples(gen, batch):
    latent_dims = gen.input.shape[-1]
    latent_points = generate_latent_points(batch, latent_dims)
    x = gen(latent_points)
    y = generate_fake_y(batch)
    return x, y


def train_critic(gen, critic, ds, batch):
    realX, realY = generate_real_samples(ds, batch)
    fakeX, fakeY = generate_fake_samples(gen, batch)

    real_loss = critic.train_on_batch(realX, realY)
    fake_loss = critic.train_on_batch(fakeX, fakeY)

    return real_loss, fake_loss


def train_gen(gan, batch):
    latent_dims = gan.input.shape[-1]
    x = generate_latent_points(batch, latent_dims)
    y = generate_real_y(batch)
    loss = gan.train_on_batch(x, y)
    return loss


def summarize_performance(gen, critic, gan, cfg, epoch):
    latent_dims = gen.input.shape[-1]
    latent_points = generate_latent_points(cfg["batch"], latent_dims)
    images = gen(latent_points)
    save_image(images[:25], f"{epoch:03d}")


def train(gen, critic, gan, cfg):
    writer = define_writer()
    critic_real_losses, critic_fake_losses, gen_losses = list(), list(), list()

    steps_per_epoch = cfg["dataset"].shape[0] // cfg["batch"]
    for epoch in range(cfg["epochs"]):
        for step in range(steps_per_epoch):

            critic_real_losses_temp, critic_fake_losses_temp = list(), list()
            for i in range(cfg["n_critic"]):
                critic_real_loss, critic_fake_loss = train_critic(gen, critic, cfg["dataset"], cfg["batch"] // 2)
                critic_real_losses_temp.append(critic_real_loss)
                critic_fake_losses_temp.append(critic_fake_loss)

            critic_real_losses.append(np.mean(critic_real_losses_temp))
            critic_fake_losses.append(np.mean(critic_fake_losses_temp))

            gen_loss = train_gen(gan, cfg["batch"])
            # gen_loss=0
            gen_losses.append(gen_loss)

            print(f"{epoch} {step:2d}/{steps_per_epoch}: critic real/fake={critic_real_losses[-1]:8.3f}/{critic_fake_losses[-1]:8.3f} {gen_loss=:8.3f}")

        summarize_performance(gen, critic, gan, cfg, epoch)

    with writer.as_default():
        for i in range(len(critic_real_losses)):
            tf.summary.scalar("critic_real_loss", critic_real_losses[i], step=i)
            tf.summary.scalar("critic_fake_loss", critic_fake_losses[i], step=i)
        for i in range(len(gen_losses)):
            tf.summary.scalar("gen_loss", gen_losses[i], step=i)

def main():
    cfg = {
        "epochs": 20,
        "batch": 64,
        "n_critic": 5,
        "dataset": load_dataset(),
        "latent_dims": 100
    }
    # save_image(cfg["dataset"][:25], "init")

    critic = define_critic(cfg["dataset"].shape[1:])
    # critic.summary(show_trainable=True)
    generator = define_generator(cfg["latent_dims"])
    # generator.summary()
    gan = define_gan(generator, critic)
    # gan.summary(show_trainable=True, expand_nested=True)

    train(generator, critic, gan, cfg)

if __name__ == "__main__":
    main()
