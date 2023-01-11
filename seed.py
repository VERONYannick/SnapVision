"""Generate images with and without snapcode"""
import os
import random
import shutil

from PIL import Image, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

SOURCE_DIR = "imgs_optimized"
TARGET_DIR = "output"
STICKER = Image.open("code_opti.png")
VALIDATION_SPLIT = None
SNAP_SPLIT = 0.5
LIMIT_IMG = 0
IMG_SIZE = 256


def expand2square(pil_img, background_color):
    """expand img size to a square"""
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def apply_sticker(image, sticker):
    """Apply a sticker on an image"""
    # Rotate the sticker
    # angle = random.uniform(-45, 45)
    # sticker = sticker.rotate(angle, expand=True)

    # Get the size of the image
    image_width, image_height = image.size
    min_size = min(image_height, image_width)
    # scale = random.uniform(0.20, 0.40)
    scale = 0.3
    size = int(min_size * scale)
    sticker = sticker.resize((size, size))

    # Get the size of the image and sticker
    image_width, image_height = image.size
    sticker_width, sticker_height = sticker.size

    # Generate a random position
    x = random.randint(0, image_width - sticker_width)
    y = random.randint(0, image_height - sticker_height)

    # Paste the sticker on the image
    image.paste(sticker, (x, y), sticker)

    return image


def build_folders():
    """Cleanup and rebuild folders structure"""
    print("Building folders structure...")
    if os.path.exists(TARGET_DIR) and os.path.isdir(TARGET_DIR):
        shutil.rmtree(TARGET_DIR)
    os.mkdir(TARGET_DIR)
    if VALIDATION_SPLIT:
        for split in ["training", "validation"]:
            os.mkdir(os.path.join(TARGET_DIR, split))
            os.mkdir(os.path.join(TARGET_DIR, split, "snap"))
            os.mkdir(os.path.join(TARGET_DIR, split, "clean"))
    else:
        os.mkdir(os.path.join(TARGET_DIR, "snap"))
        os.mkdir(os.path.join(TARGET_DIR, "clean"))
    print("Building folders complete")


def main():
    """Seed img"""
    limit_img = LIMIT_IMG  # Limit the number of images to seed (0=no limit)
    current_img = 0
    nb_img = len(os.listdir(SOURCE_DIR))

    print(f"Images founds : {nb_img}")
    if limit_img > 0 and limit_img < nb_img:
        print(f"Images founds exceed max img, limiting to {LIMIT_IMG} images")
    if limit_img < 1:
        limit_img = nb_img

    build_folders()

    files_nb = {
        "training": {"snap": 0, "clean": 0},
        "validation": {"snap": 0, "clean": 0},
    }

    pbar = tqdm(total=min(limit_img, nb_img), desc="Seeding pictures")
    for filename in os.listdir(SOURCE_DIR):
        current_img += 1
        if current_img > limit_img:
            break
        pbar.update(1)

        # Skip non-image files
        if not filename.endswith(".jpg"):
            continue

        image = Image.open(os.path.join(SOURCE_DIR, filename))
        # image.thumbnail((256,256)) #Warning : does not scale up smaller img => but speed up expand

        label = ""
        if random.random() > SNAP_SPLIT:
            image = apply_sticker(image, STICKER.copy())
            label = "snap"
        else:
            label = "clean"

        base_path = TARGET_DIR
        file_type = "training"
        if VALIDATION_SPLIT:
            if random.random() < VALIDATION_SPLIT:
                file_type = "validation"
            base_path = os.path.join(base_path, file_type)
            files_nb[file_type][label] += 1
        else:
            files_nb["training"][label] += 1

        image = expand2square(image, "black").resize((IMG_SIZE, IMG_SIZE))

        image.save(
            os.path.join(
                base_path,
                label,
                f"{label}_{files_nb[file_type][label]}{os.path.splitext(filename)[1]}",
            )
        )

    # with open(os.path.join(destinationDirectory,'infos.json'),"w") as outfile:
    #    json.dump(infos, outfile, indent=4)


if __name__ == "__main__":
    main()
