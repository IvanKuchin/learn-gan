import os.path

from matplotlib import pyplot
import numpy as np
import tensorflow as tf


def generate_latent_points(batch, latent_dims):
    points = np.random.random(batch * latent_dims) * 2 - 1
    points = np.reshape(points, (batch, latent_dims))
    return points


def generate_samples(model, batch):
    filename = "latent_math/latent_points.npy"
    if os.path.exists(filename):
        latent_points = np.load(filename)
    else:
        latent_dims = model.input.shape[1]
        latent_points = generate_latent_points(batch, latent_dims)
        np.save(filename, latent_points)
    preds = model.predict(latent_points)
    return preds


def display_images(images, rows, cols):
    min = images.min()
    max = images.max()
    images = (images - min) / (max - min)
    for i in range(rows * cols):
        pyplot.subplot(rows, cols, i+1)
        pyplot.axis(False)
        pyplot.imshow(images[i])
    pyplot.show()
    pyplot.close()


def math(preds):
    # image prep
    idx_smiling_w = np.array([1, 2, 30, 80])
    idx_normal_w = np.array([3, 24, 37, 43])
    idx_normal_m = np.array([0, 38, 93, 87])

    smiling_w = preds[idx_smiling_w]
    normal_w = preds[idx_normal_w]
    normal_m = preds[idx_normal_m]

    smiling_w = np.append(smiling_w, np.mean(smiling_w, axis=0, keepdims=True), axis=0)
    normal_w = np.append(normal_w, np.mean(normal_w, axis=0, keepdims=True), axis=0)
    normal_m = np.append(normal_m, np.mean(normal_m, axis=0, keepdims=True), axis=0)

    images = np.vstack([smiling_w, normal_w, normal_m])
    display_images(images, 3, 5)

    smiling_m = smiling_w[-1] - normal_w[-1] + normal_m[-1]
    display_images(np.expand_dims(smiling_m, 0), 1, 1)


def main():
    model = tf.keras.models.load_model("models/gen_100.h5")
    preds = generate_samples(model, 10*10)

    math(preds)


if __name__ == "__main__":
    main()
