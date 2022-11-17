import numpy as np
import tensorflow as tf
import skimage
import scipy
import tqdm


def scale_images(images, shape):
    arr = list()
    for i in tqdm.tqdm(range(images.shape[0])):
        new_img = skimage.transform.resize(images[i], shape)
        arr.append(new_img)

    return np.asarray(arr)


def calculate_fid(features1, features2):
    # calculate mean and covariance statistics
    mu1, sigma1 = features1.mean(axis=0), np.cov(features1, rowvar=False)
    mu2, sigma2 = features2.mean(axis=0), np.cov(features2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = scipy.linalg.sqrtm(sigma1@sigma2)
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def main():
    # load model
    model = tf.keras.applications.inception_v3.InceptionV3(include_top=False, pooling="avg", input_shape=(299, 299, 3))
    # model.summary()

    # prepare data
    # images1 = np.random.randint(0, 255, 10*32*32*3)
    # images1 = np.reshape(images1, (10, 32, 32, 3))
    # images2 = np.random.randint(0, 255, 10*32*32*3)
    # images2 = np.reshape(images2, (10, 32, 32, 3))
    (images1, _), (images2, _) = tf.keras.datasets.cifar10.load_data()
    np.random.shuffle(images1)
    np.random.shuffle(images2)
    images1 = images1[:1000]
    images2 = images2[:1000]
    print("Prepared images:", images1.shape, images2.shape)

    images1 = images1.astype("float32")
    images2 = images2.astype("float32")

    images1 = scale_images(images1, (299, 299, 3))
    images2 = scale_images(images2, (299, 299, 3))
    print("Scaled images:", images1.shape, images2.shape)

    images1 = tf.keras.applications.inception_v3.preprocess_input(images1)
    images2 = tf.keras.applications.inception_v3.preprocess_input(images2)

    pred1 = model.predict(images1)
    pred2 = model.predict(images2)
    print("Predictions:", pred1.shape, pred2.shape)

    print("fid same:", calculate_fid(pred1, pred1))
    print("fid diff:", calculate_fid(pred1, pred2))


if __name__ == "__main__":
    main()