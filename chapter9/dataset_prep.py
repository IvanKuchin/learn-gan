from tqdm import tqdm
import datetime
import os
import numpy as np
from PIL import Image
from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN


def load_image(filename):
    arr = np.array([])
    with Image.open(filename) as img:
        rgb = img.convert("RGB")
        arr = np.asarray(rgb)

    return arr


def load_images(dir):
    images = []
    for i, file in tqdm(enumerate(os.listdir(dir))):
        image = load_image(f"{dir}/{file}")
        images.append(image)
        if i == 16 - 1:
            break

    return np.asarray(images)


def show_images(img):
    cardinality = len(img.shape)
    if cardinality == 4:
        count = img.shape[0]
        count_sqrt = int(np.sqrt(count))
        for i in range(count_sqrt**2):
            pyplot.subplot(count_sqrt, count_sqrt, i+1)
            pyplot.axis(False)
            pyplot.imshow(img[i])
        pyplot.show()
        pyplot.close()
    else:
        print(f"ERROR: images cardinality must be 4, but given {img.shape}")


def crop_and_resize(images):
    result = []
    cardinality = len(images.shape)
    if cardinality == 4:
        model = MTCNN()
        for i in range(images.shape[0]):
            faces = model.detect_faces(images[i])
            if len(faces) == 0:
                continue
            (x1, y1, width, height) = faces[0]["box"]
            x2 = x1 + width
            y2 = y1 + height
            face = images[i, y1:y2, x1:x2, :]

            img = Image.fromarray(face)
            img = img.resize((80, 80))
            img = img.convert("RGB")
            result.append(np.asarray(img))

            print(f"{i}, {face.shape}")
    else:
        print("Cardinality must be 4, but given shape is {images.shape")

    return np.asarray(result)


def generate_dataset(dir, dst):
    print("loading images ...")
    images = load_images(dir)
    print(f"loaded images shape is {images.shape}")
    # show_images(images)
    print("crop and resizing faces ...")
    face_images = crop_and_resize(images)
    print(f"cropped images shape is {images.shape}")
    # show_images(face_images)
    print("saving images to a file ...")
    start = datetime.datetime.now()
    np.save(dst, face_images)
    print(datetime.datetime.now() - start)
    return


def main():
    src_dir = "img_align_celeba"
    ds_file = "dataset/celeba"
    if not(os.path.isfile(f"{ds_file}.npy")):
        generate_dataset(dir=src_dir, dst=ds_file)
    else:
        print("dataset has already been built")
        start = datetime.datetime.now()
        ds = np.load(f"{ds_file}.npy")
        load_time = datetime.datetime.now() - start
        print(f"{load_time}, shape: {ds.shape}")


if __name__ == '__main__':
    main()