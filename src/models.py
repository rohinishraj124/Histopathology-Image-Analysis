"""
Model architectures: the preserved Xception+MobileNetV2 ensemble baseline,
a modern EfficientNetB0 transfer-learning baseline, and a U-Net segmentation
*plan* (architecture only — not trained; this dataset has no mask labels).

INPUT CONVENTION: all builders here expect [0, 255] float32 input (see
src/preprocessing.py). Each builder applies its own required scaling internally
as its first step: Xception/MobileNetV2 via their respective preprocess_input
(-> [-1, 1]), EfficientNetB0 via its built-in Rescaling layer (expects [0, 255]
directly - do NOT pre-normalize, or you'll double-rescale and starve the
pretrained features of signal).
"""
import tensorflow as tf
from tensorflow.keras.applications import Xception, MobileNetV2, EfficientNetB0
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.layers import (
    Input, GlobalAveragePooling2D, Concatenate, Dropout, Dense, Lambda,
    Conv2D, MaxPooling2D, Conv2DTranspose, concatenate,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def _compile(model, learning_rate):
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def build_ensemble_model(image_size=96, learning_rate=1e-4):
    """Preserved from the original notebook (Xception + MobileNetV2 concatenation),
    modernized: tensorflow.keras imports, learning_rate= instead of lr=/decay=,
    and each branch now applies its own required preprocess_input scaling on the
    shared [0, 255] input (this was missing before and left both backbones seeing
    out-of-distribution inputs relative to their ImageNet training)."""
    input_shape = (image_size, image_size, 3)
    inputs = Input(input_shape)

    xception_input = Lambda(xception_preprocess, name="xception_preprocess")(inputs)
    mobilenet_input = Lambda(mobilenet_preprocess, name="mobilenet_preprocess")(inputs)

    xception = Xception(include_top=False, weights="imagenet", input_shape=input_shape)(xception_input)
    mobile_net = MobileNetV2(include_top=False, weights="imagenet", input_shape=input_shape)(mobilenet_input)

    features = Concatenate(axis=-1)(
        [GlobalAveragePooling2D()(xception), GlobalAveragePooling2D()(mobile_net)]
    )
    x = Dropout(0.5)(features)
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs, name="xception_mobilenet_ensemble")
    return _compile(model, learning_rate)


def build_efficientnet_model(image_size=96, learning_rate=1e-4, fine_tune_at=None):
    """Modern transfer-learning baseline added alongside the preserved ensemble.
    Expects [0, 255] input directly - EfficientNetB0 rescales internally."""
    input_shape = (image_size, image_size, 3)
    base = EfficientNetB0(include_top=False, weights="imagenet", input_shape=input_shape)
    base.trainable = fine_tune_at is not None
    if fine_tune_at is not None:
        for layer in base.layers[:fine_tune_at]:
            layer.trainable = False

    inputs = Input(input_shape)  # [0, 255] float32
    x = base(inputs)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs, name="efficientnet_b0")
    return _compile(model, learning_rate)


def build_unet(input_shape=(96, 96, 3)):
    """Segmentation ARCHITECTURE PLAN ONLY. Not trained anywhere in this project —
    the Kaggle Histopathologic Cancer Detection dataset has no pixel-level mask
    labels. Ready to train once CAMELYON-style (image, mask) pairs are available."""

    def conv_block(x, filters):
        x = Conv2D(filters, 3, activation="relu", padding="same")(x)
        x = Conv2D(filters, 3, activation="relu", padding="same")(x)
        return x

    inputs = Input(input_shape)

    c1 = conv_block(inputs, 32)
    p1 = MaxPooling2D()(c1)
    c2 = conv_block(p1, 64)
    p2 = MaxPooling2D()(c2)
    c3 = conv_block(p2, 128)
    p3 = MaxPooling2D()(c3)

    b = conv_block(p3, 256)

    u3 = Conv2DTranspose(128, 2, strides=2, padding="same")(b)
    u3 = concatenate([u3, c3])
    d3 = conv_block(u3, 128)

    u2 = Conv2DTranspose(64, 2, strides=2, padding="same")(d3)
    u2 = concatenate([u2, c2])
    d2 = conv_block(u2, 64)

    u1 = Conv2DTranspose(32, 2, strides=2, padding="same")(d2)
    u1 = concatenate([u1, c1])
    d1 = conv_block(u1, 32)

    outputs = Conv2D(1, 1, activation="sigmoid")(d1)
    return Model(inputs, outputs, name="unet_segmentation_plan")
