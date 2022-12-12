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
    for i in range(edge**2):
        pyplot.subplot(edge, edge, i+1)
        pyplot.axis(False)
        pyplot.imshow(images[i])
    pyplot.savefig(f"progress/{name}.png")
    pyplot.close()


def prep():
    if not os.path.isdir("progress"):
        os.mkdir("progress")


def define_discriminator(image_shape):
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

    model = tf.keras.Model(inputs=inputs, outputs=patch_out, name="discriminator")
    opt = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss="mse", optimizer=opt, loss_weights=[0.5])
    return model


def main():
    prep()
    cfg = {
        "batch": 1,
        "epochs": 100,
        "dataset": load_dataset()
    }

    image_shape = cfg["dataset"][0].shape[1:]
    disc_A = define_discriminator(image_shape)
    disc_B = define_discriminator(image_shape)

    tf.keras.utils.plot_model(disc_A, f"progress/disc_A.png", show_shapes=True)

if __name__ == "__main__":
    main()
