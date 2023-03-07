Detect stickers on images using Tensorflow

# Scripts and functions

## gen_dataset.py

Download, extract and resize images to use has dataset.
Can handle multiple urls of `.zip` or `.tar` files

Download are NOT parallelized and files are extracted and resized one by one to reduce disk usage (increase time needed)

## opti_img.py

Only resize images (if you already have a local dataset)

## seed.py

Apply a sticker on a fraction of dataset and create folders of images with and without a sticker  
Create picture with sticker by pasting a png on the picture at random (scale, rotation, position)

## train.py

Build and train a model on the previously build dataset  
Save the model in `h5` format

## control.py

Load a `h5` model and test it with the validation dataset


# To run on AMD GPU

```bash
pip install tensorflow=2.10
pip install tensorflow-directml-plugin
```