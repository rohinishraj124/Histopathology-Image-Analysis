"""
Reusable inference entrypoint: Image -> Preprocessing -> Model -> Prediction ->
Confidence -> (optional) Grad-CAM. Called by app/app.py and usable standalone.
"""
import cv2
import numpy as np

from src.explainability import make_gradcam_heatmap, overlay_gradcam


def predict_image(image_path, model, image_size=96, last_conv_layer_name=None):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (image_size, image_size))
    img_model_input = img_resized.astype("float32")  # [0, 255] - model rescales internally, see src/models.py
    batch = np.expand_dims(img_model_input, axis=0)

    prob = float(model.predict(batch, verbose=0)[0, 0])
    pred_class = "tumor" if prob >= 0.5 else "no_tumor"
    confidence = prob if pred_class == "tumor" else 1 - prob

    result = {
        "image_path": image_path,
        "predicted_class": pred_class,
        "tumor_probability": round(prob, 4),
        "confidence": round(confidence, 4),
    }

    if last_conv_layer_name is not None:
        heatmap = make_gradcam_heatmap(batch, model, last_conv_layer_name)
        img_display = img_resized.astype("float32") / 255.0  # overlay_gradcam expects [0, 1] for display
        result["gradcam_overlay"] = overlay_gradcam(img_display, heatmap)

    return result
