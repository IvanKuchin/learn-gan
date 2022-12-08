import os

import numpy as np
import tensorflow as tf
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


def convert_dataset(dir, size):
    sat, gmap = list(), list()
    for file in os.listdir(dir):
        img = tf.keras.utils.load_img(f"{dir}/{file}", target_size=size)
        arr = tf.keras.utils.img_to_array(img)
        sat.append(arr[:, :256, ...])
        gmap.append(arr[:, 256:, ...])
    return np.asarray(sat), np.asarray(gmap)


def save(ds, name):
    np.save(name, ds)


def main():
    train = convert_dataset("dataset/train", (256, 512))
    save_images(train[0][:25], "ds_sat")
    save_images(train[1][:25], "ds_map")
    print(f"satellite shape: {train[0].shape}")
    print(f"map shape: {train[1].shape}")
    save(train, "train")
    return

if __name__ == "__main__":
    main()
