import os
import shutil


def split_files(src_dir, dst_dir, files_per_dir):
    files = os.listdir(src_dir)
    current_split = 0

    for i, file in enumerate(files):
        if i % files_per_dir == 0:
            current_split += 1
            print("split %02d" % current_split)
            os.mkdir(f"{dst_dir}/{current_split}")
        shutil.copy(f"{src_dir}/{file}", f"{dst_dir}/{current_split}")
    return


def main():
    split_files(src_dir="dataset_images",
                dst_dir="dataset_images_splits",
                files_per_dir=3000)


if __name__ == "__main__":
    main()