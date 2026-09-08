"""
tf.data preprocessing pipeline. Replaces the original notebook's
ImageDataGenerator + shutil.copyfile-into-class-folders approach — this builds
batches directly from a dataframe of (id, label) pairs with no on-disk copy step.

IMPORTANT (pixel scaling): images are kept in their native [0, 255] float32 range
(NOT normalized to [0, 1]) because Xception, MobileNetV2, and EfficientNetB0 each
require different input scaling, and EfficientNetB0 specifically has rescaling
built into the model itself. Each model in src/models.py applies its own correct
scaling as its first layer. Do not add a /255.0 normalization here.

IMPORTANT (file format): this dataset ships .tif images. TensorFlow's native
tf.image.decode_image does NOT support TIFF (only JPEG/PNG/GIF/BMP/WebP) and will
fail with an InvalidArgumentError at graph-execution time. PIL.Image.open handles
TIFF natively, so it's used here via tf.py_function to plug into the tf.data graph.
Trade-off: tf.py_function runs per-sample in eager mode on CPU, so it's slower than
a pure graph-mode decode op - acceptable since image loading is I/O-bound anyway,
but worth knowing if throughput ever becomes a bottleneck.
"""
import os

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom, RandomContrast

augment = tf.keras.Sequential(
    [
        RandomFlip("horizontal_and_vertical"),
        RandomRotation(0.5),
        RandomZoom(0.1),
        RandomContrast(0.1),
    ],
    name="augmentation",
)


def _load_tif(path, image_size):
    path = path.numpy().decode("utf-8")
    img = Image.open(path).convert("RGB")
    img = img.resize((image_size, image_size))
    return np.array(img, dtype=np.float32)  # native [0, 255] - see module docstring


def load_and_preprocess(path, label, image_size=96):
    img = tf.py_function(func=lambda p: _load_tif(p, image_size), inp=[path], Tout=tf.float32)
    img.set_shape([image_size, image_size, 3])  # py_function loses shape info - restore explicitly
    return img, label


def make_dataset(dataframe, image_dir, training, batch_size=64, image_size=96, seed=42):
    paths = (dataframe["id"] + ".tif").apply(lambda f: os.path.join(image_dir, f)).values
    labels = dataframe["label"].values.astype("float32")

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(buffer_size=len(dataframe), seed=seed)

    ds = ds.map(
        lambda p, l: load_and_preprocess(p, l, image_size), num_parallel_calls=tf.data.AUTOTUNE
    )
    ds = ds.batch(batch_size)
    if training:
        ds = ds.map(lambda x, y: (augment(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)
