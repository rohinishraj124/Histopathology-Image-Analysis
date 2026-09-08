"""
Grad-CAM explainability. See notebooks/histopathology_analysis.ipynb for the
discussion of what Grad-CAM does and does not tell us about model correctness.
"""
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model


def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = Model(model.inputs, [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_gradcam(image_rgb_float, heatmap, alpha=0.4):
    heatmap_resized = cv2.resize(heatmap, (image_rgb_float.shape[1], image_rgb_float.shape[0]))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB) / 255.0
    return alpha * heatmap_color + (1 - alpha) * image_rgb_float
