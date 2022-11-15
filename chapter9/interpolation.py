import numpy as np
import tensorflow as tf
from matplotlib import pyplot


def generate_latent_points(batch, latent_dims):
    points = np.random.random(batch * latent_dims) * 2 - 1
    points = np.reshape(points, (batch, latent_dims))
    return points


def generate_samples(gen, batch):
    latent_dims = gen.input.shape[1]
    latent_points = generate_latent_points(batch, latent_dims)
    preds = gen.predict(latent_points)
    return preds


def draw_images(images, rows, cols):
    min = images.min()
    max = images.max()
    images = (images - min) / (max - min)
    for i in range(rows * cols):
        pyplot.subplot(rows, cols, i + 1)
        pyplot.axis(False)
        pyplot.imshow(images[i])
    pyplot.show()
    pyplot.close()


def interpolate_pair(images):
    steps = 10
    ratios = np.linspace(0, 1, steps)
    a = images[0]
    b = images[1]
    arr = list()
    for ratio in ratios:
        arr.append(a * (1-ratio) + b * ratio)
    return np.asarray(arr)


def interpolate_pairs(images):
    lst = list()
    for i in range(0, images.shape[0], 2):
        images_row = interpolate_pair(images[i:i+2])
        lst.append(images_row)
    arr = np.vstack(lst)
    return arr


def main():
    rows = 6
    gen = tf.keras.models.load_model("models/gen_090.h5")
    preds = generate_samples(gen, rows * 2)
    interpolations = interpolate_pairs(preds)
    draw_images(interpolations, rows, interpolations.shape[0] // rows)


if __name__ == "__main__":
    main()
