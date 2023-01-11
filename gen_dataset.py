"""Load, extract and optimize images from a zip url"""
import os
import sys
import tarfile
from enum import Enum
from zipfile import ZipFile

import requests
from PIL import Image, ImageFile
from tqdm import tqdm

URLS = []  # Your dataset urls here


class GenModes(Enum):
    """Generation mode"""

    OVERRIDE = 1
    APPEND = 2


ImageFile.LOAD_TRUNCATED_IMAGES = True

ARCHIVE_FILE = "images.zip"
TARGET_DIR = "imgs_optimized"
LIMIT_IMG = 0
GEN_MODE = GenModes.OVERRIDE


def download_url(dl_url, save_path, chunk_size=128):
    """download an archive at save_path"""
    response = requests.get(dl_url, stream=True, timeout=1000)
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    progress_bar = tqdm(
        total=total_size_in_bytes,
        unit="iB",
        unit_scale=True,
        desc="Downloading archive",
        position=1,
        leave=True,
    )
    with open(save_path, "wb") as output_file:
        for chunk in response.iter_content(chunk_size=chunk_size):
            progress_bar.update(len(chunk))
            output_file.write(chunk)
    progress_bar.close()


def extract_zip(img_nb):
    """extract zip file"""
    with ZipFile(ARCHIVE_FILE) as archive:
        filenames = list(
            filter(lambda filename: filename.endswith(".JPG"), archive.namelist())
        )
        limit_img = LIMIT_IMG or len(filenames)
        pbar = tqdm(total=limit_img, desc="Processing images", position=1, leave=True)
        for filename in filenames:
            with archive.open(filename) as file:
                img_nb += 1
                handle_file(file, filename, img_nb)
                pbar.update(1)


def extract_tar(img_nb):
    """extract tar file"""
    with tarfile.open("./imdb_0.tar", "r") as tar_file:
        files_info = list(
            filter(
                lambda filename: filename.name.endswith("jpg"), tar_file.getmembers()
            )
        )
        limit_img = LIMIT_IMG or len(files_info)
        pbar = tqdm(total=limit_img, desc="Processing images", position=1, leave=True)
        # tar_file.extractall(members=[x for x in tar.getmembers() if x.name in files_i_want])
        for file_info in files_info:
            img_nb += 1
            file = tar_file.extractfile(file_info.name)
            handle_file(file, file_info.name, img_nb)
            pbar.update(1)


def handle_file(file, filename, img_nb):
    """Save file"""
    img = Image.open(file)
    img.thumbnail((256, 256))  # Warning : does not scale up smaller img
    img.save(
        os.path.join(
            TARGET_DIR,
            f"img_{img_nb}{os.path.splitext(filename)[1]}",
        )
    )


def main():
    """load images"""
    img_nb = 0
    filelist = os.listdir(TARGET_DIR)
    if GEN_MODE == GenModes.OVERRIDE:
        for file in tqdm(filelist, desc="Deleting old pictures"):
            os.remove(os.path.join(TARGET_DIR, file))
    elif GEN_MODE == GenModes.APPEND:
        img_nb = len(filelist) - 1
    else:
        print("Invalid GEN_MODE")
        sys.exit(1)

    for url in tqdm(URLS, desc="Downloading from urls", position=0):
        download_url(url, ARCHIVE_FILE)
        if ARCHIVE_FILE.endswith(".zip"):
            extract_zip(img_nb)
        elif ARCHIVE_FILE.endswith(".tar"):
            extract_tar(img_nb)
        else:
            print("Unsuported archive type")
            sys.exit(1)

        os.remove(ARCHIVE_FILE)
    sys.exit(0)


if __name__ == "__main__":
    main()
