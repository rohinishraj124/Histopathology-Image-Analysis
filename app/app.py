"""
Minimal Gradio demo for the histopathology patch classifier.

Run:
    python app/app.py --model-path ./models/efficientnet_best.keras

This is a research/educational prototype UI, not a diagnostic device.
"""
import argparse
import sys
import os

import cv2
import gradio as gr
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.inference import predict_image  # noqa: E402


def build_app(model, last_conv_layer_name):
    def gradio_predict(image):
        tmp_path = "/tmp/_gradio_input.png"
        cv2.imwrite(tmp_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        result = predict_image(tmp_path, model, last_conv_layer_name=last_conv_layer_name)
        overlay = result.get("gradcam_overlay")
        return result["predicted_class"], f"{result['confidence']:.2%}", overlay

    demo = gr.Interface(
        fn=gradio_predict,
        inputs=gr.Image(label="Histopathology patch (96x96 H&E)"),
        outputs=[
            gr.Text(label="Prediction"),
            gr.Text(label="Confidence"),
            gr.Image(label="Grad-CAM overlay"),
        ],
        title="Histopathology Metastatic Cancer Detection — Research Prototype",
        description=(
            "Upload a small H&E-stained tissue patch. This is a research/educational "
            "prototype trained on the PatchCamelyon-derived Kaggle dataset — it is NOT "
            "a diagnostic device and has not been clinically validated."
        ),
    )
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--last-conv-layer", default="top_conv",
                         help="Name of the last conv layer for Grad-CAM (depends on the model used).")
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model_path)
    app = build_app(model, args.last_conv_layer)
    app.launch()
