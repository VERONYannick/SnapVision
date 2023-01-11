"""Build and train NN model"""
import os
import sys
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
import sklearn.model_selection
import tensorflow as tf
from PIL import Image
from tqdm import tqdm

# Set the path to the directory containing the image files
IMAGE_DIR = "Output"
LIMIT = 2000  # Limit dataset size (0=no limit) (will load this limit for each category)
EPOCH = 10
STEP_EPOCH = 200

IMG_SIZE = 256

TRAINING_DATASET = False
FULL_STEP_NB = 10


class LoadModes(Enum):
    """Loading mode"""

    FULL = 1
    FULL_STEP = 2
    GENERATOR = 3


LOADING_MODE = LoadModes.FULL_STEP


def load_data(offset=0):
    """Load data in memory"""
    # Create a list of file paths
    file_paths = [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR)]

    nb_img_clean = len(os.listdir(os.path.join(IMAGE_DIR, "clean")))
    nb_img_snap = len(os.listdir(os.path.join(IMAGE_DIR, "snap")))
    print(f"{nb_img_clean} clean, {nb_img_snap} snap datas founds")

    if LIMIT > 0 and LIMIT < nb_img_clean:
        print(f"clean data found exceed maximum, limiting to {LIMIT}")
        nb_img_clean = LIMIT
    if LIMIT > 0 and LIMIT < nb_img_snap:
        print(f"snap data found exceed maximum, limiting to {LIMIT}")
        nb_img_snap = LIMIT

    images = np.empty((nb_img_clean + nb_img_snap, IMG_SIZE, IMG_SIZE, 3), dtype=int)
    labels = np.empty(nb_img_clean + nb_img_snap, dtype=int)

    print(np.shape(images))

    # Read the images and labels from the file paths
    curr_img = 0
    current_img_by_type = 0
    loaded_snap = 0
    loaded_clean = 0
    pbar = tqdm(total=nb_img_clean + nb_img_snap, leave=True, desc="Loading dataset")
    for file_path in file_paths:
        filenames = os.listdir(file_path)[offset:]
        if len(filenames) == 0:
            print(f"Dataset to small for offset {offset}")
            exit(1)
        for filename in filenames:
            current_img_by_type += 1
            if LIMIT > 0 and current_img_by_type > LIMIT:
                current_img_by_type = 0
                break
            curr_img += 1
            pbar.update(1)
            # Load the image
            image = tf.keras.preprocessing.image.load_img(
                os.path.join(file_path, filename)
            )

            # Convert the image to a numpy array
            image = tf.keras.preprocessing.image.img_to_array(image)
            images[curr_img - 1] = image
            if file_path.split("\\")[1] == "clean":
                labels[curr_img - 1] = 0
                loaded_clean += 1
            else:
                labels[curr_img - 1] = 1
                loaded_snap += 1
            # Add the image and label to the lists

        # print(np.shape(image))
        # images.append(image)
        # if(file_path.split('\\')[1] == "clean"):
        #   labels.append(0)
        # else:
        #   labels.append(1)

    pbar.close()
    print(f"{loaded_clean} clean img loaded, {loaded_snap} snap img loaded")

    # images = np.array(images)
    # labels = np.array(labels)
    # print(type(images))

    # Preprocess the data
    images = tf.keras.utils.normalize(images, axis=1)
    print("Data normalized")

    # Split the data into train, validation, and test sets
    x_train_f, x_test_f, y_train_f, y_test_f = sklearn.model_selection.train_test_split(
        images, labels, test_size=0.2, shuffle=True
    )
    x_train_f, x_val_f, y_train_f, y_val_f = sklearn.model_selection.train_test_split(
        x_train_f, y_train_f, test_size=0.2, shuffle=True
    )

    return (x_train_f, y_train_f, x_val_f, y_val_f, x_test_f, y_test_f)


# create augmentation
# TODO try with augmentation (/!\ may also augment validation data)
def load_data_gen():
    """load data using a generator"""

    if TRAINING_DATASET:
        training_path = os.path.join(IMAGE_DIR, "training")
        validation_path = os.path.join(IMAGE_DIR, "validation")

        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1.0 / 255,  # prevent out of range
        )
        test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1.0 / 255,
        )
        train_gen = train_datagen.flow_from_directory(
            training_path,
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=100,
            class_mode="categorical",
        )
        valid_gen = test_datagen.flow_from_directory(
            validation_path,
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=100,
            class_mode="categorical",
        )
        return (train_gen, valid_gen)

    else:
        image_flow = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1.0 / 255, validation_split=0.2
        )
        train_gen = image_flow.flow_from_directory(
            IMAGE_DIR,
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=100,
            subset="training",
            class_mode="categorical",
            # shuffle=True,
        )  # save_to_dir augmentation possible
        valid_gen = image_flow.flow_from_directory(
            IMAGE_DIR,
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=100,
            subset="validation",
            class_mode="categorical",
            # shuffle=True,
        )  # save_to_dir augmentation possible
        return (train_gen, valid_gen)


def build_model():
    """Build model"""

    # print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    # print(tf.config.list_physical_devices('GPU'))
    # print(tf.config.list_physical_devices())
    # exit()

    # Build the model

    # model = tf.keras.Sequential([
    #   tf.keras.layers.Flatten(input_shape=(256, 256, 3)),
    #   tf.keras.layers.Dense(128, activation='relu'),
    #   tf.keras.layers.Dropout(0.2),
    #   tf.keras.layers.Dense(10, activation='softmax')
    # ])

    model = tf.keras.Sequential(
        [
            # tf.keras.layers.Rescaling(1./255),
            tf.keras.layers.Conv2D(
                32, (3, 3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)
            ),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    print("Model built")

    # Compile the model
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    print("Model compiled")

    print(model.summary())
    return model


def train_model(model):
    """Train the model"""
    if LOADING_MODE == LoadModes.FULL:
        (x_train, y_train, x_val, y_val, x_test, y_test) = load_data()
        # Train the model
        model.fit(
            x_train,
            y_train,
            epochs=EPOCH,
            steps_per_epoch=STEP_EPOCH,
            shuffle=True,
            validation_data=(x_val, y_val),
        )  # Full data
        print("Evaluating")
        # Evaluate the model
        model.evaluate(x_val, y_val)
    elif LOADING_MODE == LoadModes.FULL_STEP:
        for x in tqdm(range(FULL_STEP_NB), desc="Full step running"):
            (x_train, y_train, x_val, y_val, x_test, y_test) = load_data(x * LIMIT)
            model.fit(
                x_train,
                y_train,
                epochs=EPOCH,
                steps_per_epoch=STEP_EPOCH,
                shuffle=True,
                validation_data=(x_val, y_val),
            )
        print("Evaluating")
        # Evaluate the model
        model.evaluate(x_val, y_val)

    elif LOADING_MODE == LoadModes.GENERATOR:
        (train_gen, valid_gen) = load_data_gen()

        if False:
            for x in train_gen:
                # print(np.shape(x))
                print(np.shape(x[0]))
                print(np.shape(x[0][0]))
                # img = Image.fromarray(x[0][0], "RGB")
                # plt.imshow(img)
                plt.imshow(x[0][0])
                plt.show()
                # print(np.shape(x))
                print(x[0])
                print(np.shape(x[1]))
                print(x[1])

        print(train_gen.class_indices)

        model.fit(
            train_gen,
            epochs=EPOCH,
            steps_per_epoch=STEP_EPOCH,
            validation_data=valid_gen,
            validation_steps=50,
        )
        print("Evaluating")
        # Evaluate the model
        model.evaluate(valid_gen)
    else:
        print("Unsuported loading mode")
        sys.exit(1)
    return model


def main():
    """Train model"""
    if LIMIT * 2 < EPOCH * STEP_EPOCH:
        print(
            f"EPOCH * STEP_EPOCH can't be higher than full dataset : {EPOCH}*{STEP_EPOCH} > {LIMIT*2}"
        )
        sys.exit(1)

    model = build_model()
    model = train_model(model)

    model.save("models/modelTest.h5")
    model.save_weights("models/modelTestW")


if __name__ == "__main__":
    main()
