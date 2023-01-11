"""Control model"""
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import tqdm

MODEL_DIR = "models"
DEBUG = False
# Set the path to the directory containing the image files
IMAGE_DIR = "Output"
LIMIT = 1000  # Limit dataset size (0=no limit)
IMG_SIZE = 256


def main():
    """control model quality"""
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
    print("Loading dataset...")
    pbar = tqdm(total=nb_img_clean + nb_img_snap)
    for file_path in file_paths:
        for filename in os.listdir(file_path):
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
            else:
                labels[curr_img - 1] = 1
            # Add the image and label to the lists

        # print(np.shape(image))
        # images.append(image)
        # if(file_path.split('\\')[1] == "clean"):
        #   labels.append(0)
        # else:
        #   labels.append(1)

    pbar.close()
    print("Dataset loaded")

    # images = np.array(images)
    # labels = np.array(labels)
    # print(type(images))

    # Preprocess the data
    images = tf.keras.utils.normalize(images, axis=1)
    print("Data normalized")

    saved_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "modelTest.h5"))
    # savedModel = model.load_weights('gfgModelWeights.h5')
    print(saved_model.summary())
    loss, acc = saved_model.evaluate(images, labels, verbose=2)
    print(
        f"Restored model, loss: {100*loss:5.2f}%, accuracy: {100*acc:5.2f}% on {nb_img_clean} clean, {nb_img_snap} snap datas"
    )

    samples_to_predict = []
    for sample in images:
        # Add sample to array for prediction
        samples_to_predict.append(sample)

    samples_to_predict = np.array(samples_to_predict)
    print(samples_to_predict.shape)

    # Generate predictions for samples
    predictions = saved_model.predict(samples_to_predict)
    formated_prediction = predictions.flatten()
    print(predictions)
    print(formated_prediction)
    # classes = np.argmax(predictions, axis = 1)
    str_label = np.where(labels == 0, "clean", "snap")
    if DEBUG:
        for x in range(len(predictions)):  # press q to close
            result = "Unsure"
            confidence = 0
            color = "red"
            if formated_prediction[x] <= 0.5:
                result = "clean"
                confidence = 1 - formated_prediction[x]
            elif formated_prediction[x] > 0.5:
                result = "snap"
                confidence = formated_prediction[x]
            if result == str_label[x]:
                color = "green"
            plt.text(
                0,
                10,
                f"{result}({str_label[x]}) confidence {confidence:.0%}",
                color=color,
                size="large",
            )
            plt.imshow(images[x])
            plt.show()

    stats = {"snap_ok": 0, "snap_bad": 0, "clean_ok": 0, "clean_bad": 0}
    wrong = {"snap": [], "clean": []}
    for x in range(len(predictions)):
        result = "snap"
        if formated_prediction[x] <= 0.5:
            result = "clean"

        if result != str_label[x]:
            if str_label[x] == "clean":
                wrong["clean"].append(images[x])
                stats["clean_bad"] += 1
            else:
                wrong["snap"].append(images[x])
                stats["snap_bad"] += 1
        else:
            if str_label[x] == "clean":
                stats["clean_ok"] += 1
            else:
                stats["snap_ok"] += 1
    print(
        f"Results => snap_ok: {stats['snap_ok']}, snap_bad: {stats['snap_bad']}, clean_ok: {stats['clean_ok']}, clean_bad: {stats['clean_bad']}"
    )
    print(
        f"Accuracy => snap : {(stats['snap_ok']/(stats['snap_ok']+stats['snap_bad']))*100:5.2f}%, clean : {(stats['clean_ok']/(stats['clean_ok']+stats['clean_bad']))*100:5.2f}% "
    )
    if stats["snap_bad"] > 0 or stats["clean_bad"] > 0:
        print("Errors snap : ")
        for err_snap in wrong["snap"]:
            plt.imshow(err_snap)
            plt.show()

        print("Errors clean : ")
        for err_clean in wrong["clean"]:
            plt.imshow(err_clean)
            plt.show()


if __name__ == "__main__":
    main()
