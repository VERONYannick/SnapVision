"""Reduce images size"""
import os
import shutil

from PIL import Image, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

SOURCE_DIR = "samples"
TARGET_DIR = "imgs_optimized"
TARGET_SIZE = 256


def main():
    """Reduce img size"""
    if os.path.exists(TARGET_DIR) and os.path.isdir(TARGET_DIR):
        shutil.rmtree(TARGET_DIR)
    os.mkdir(TARGET_DIR)

    limit_img = 0  # Limit the number of images to seed (0=no limit)
    current_img = 0
    nb_img = 0
    for folder in os.listdir(SOURCE_DIR):
        nb_img += len(os.listdir(os.path.join(SOURCE_DIR, folder)))
    print(f"Images founds : {nb_img}")
    if limit_img > 0 and limit_img < nb_img:
        print(f"Images founds exceed max img, limiting to {limit_img} images")
    if limit_img < 1:
        limit_img = nb_img

    pbar = tqdm(total=min(limit_img, nb_img))
    for folder in os.listdir(SOURCE_DIR):
        for filename in os.listdir(os.path.join(SOURCE_DIR, folder)):
            current_img += 1
            pbar.update(1)
            if current_img > limit_img:
                break

            # Skip non-image files
            if not filename.endswith(".jpg"):
                continue

            image = Image.open(os.path.join(SOURCE_DIR, folder, filename))
            image.thumbnail(
                (TARGET_SIZE, TARGET_SIZE)
            )  # Warning : does not scale up smaller img
            image.save(os.path.join(TARGET_DIR, filename))


if __name__ == "__main__":
    main()
