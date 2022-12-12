import os

import numpy as np
import tensorflow as tf
from matplotlib import pyplot


def save_images(images, name):
    cardinality = len(images.shape)
    if cardinality != 4:
        print(f"images cardinality must be 4, but given shape {images.shape}")
        return
    min, max = images.min(), images.max()
    images = (images - min) / (max - min)
    edge = int(np.sqrt(images.shape[0]))
    for i in range(edge**2):
        pyplot.subplot(edge, edge, i+1)
        pyplot.axis(False)
        pyplot.imshow(images[i])
    pyplot.savefig(f"progress/{name}.png")
    pyplot.close()


def load_images(path):
    images = list()
    for file in os.listdir(path):
        img = tf.keras.utils.load_img(f"{path}/{file}", target_size=(256, 256))
        arr = tf.keras.utils.img_to_array(img)
        images.append(arr)
    return np.asarray(images)


def prep():
    if not os.path.isdir("progress"):
        os.mkdir("progress")


def main():
    prep()
    zebras = load_images("dataset/zebras")
    horses = load_images("dataset/horses")
    np.save("dataset", [horses, zebras])

    (horses, zebras) = np.load("dataset.npy", allow_pickle=True)
    print(f"zebras {zebras.shape}, horses {horses.shape}")
    save_images(zebras[:25], "zebras")
    save_images(horses[:25], "horses")

if __name__ == "__main__":
    main()
