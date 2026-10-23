import argparse
import json
import os
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

from data_preprocessing import EMOTION_LABELS


def load_class_mapping(class_map_path: str = "models/class_mapping.json") -> Dict[int, str]:
    """Load class mapping from JSON, fallback to default mapping if absent."""
    if os.path.exists(class_map_path):
        with open(class_map_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {int(k): str(v) for k, v in raw.items()}

    print(
        "Warning: class_mapping.json not found. "
        "Using default FER2013 class mapping from source code."
    )
    return EMOTION_LABELS


def preprocess_image(image_path: str) -> Tuple[np.ndarray, Image.Image]:
    """Load image, convert to grayscale, resize to 48x48, normalize, and batch."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    try:
        img = Image.open(image_path).convert("L")
    except UnidentifiedImageError as exc:
        raise ValueError("Uploaded file is not a valid image.") from exc

    display_img = img.copy()
    img = img.resize((48, 48))

    arr = np.array(img, dtype=np.float32)
    if arr.shape != (48, 48):
        raise ValueError("Preprocessed image shape is invalid. Expected 48x48.")

    arr = arr / 255.0
    arr = np.expand_dims(arr, axis=-1)
    arr = np.expand_dims(arr, axis=0)

    return arr, display_img


def predict_emotion(
    model_path: str,
    image_path: str,
    class_map_path: str = "models/class_mapping.json",
):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}. Train the model before prediction."
        )

    class_map = load_class_mapping(class_map_path)
    model = tf.keras.models.load_model(model_path)

    image_tensor, display_img = preprocess_image(image_path)
    probs = model.predict(image_tensor, verbose=0)[0]
    pred_id = int(np.argmax(probs))
    pred_label = class_map.get(pred_id, f"Class {pred_id}")
    confidence = float(probs[pred_id])

    print("Predicted Emotion:", pred_label)
    print("Confidence:", round(confidence * 100, 2), "%")
    print("\nAll class probabilities:")
    for class_id, class_name in class_map.items():
        print(f"  {class_name}: {probs[class_id] * 100:.2f}%")

    plt.figure(figsize=(4, 4))
    plt.imshow(display_img, cmap="gray")
    plt.title(f"Prediction: {pred_label} ({confidence * 100:.2f}%)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def colab_upload_demo(
    model_path: str = "models/best_emotion_model.keras",
    class_map_path: str = "models/class_mapping.json",
):
    """Run upload-based prediction demo in Google Colab."""
    try:
        from google.colab import files
    except Exception as exc:
        raise EnvironmentError(
            "This upload demo is designed for Google Colab. "
            "Use predict_emotion() directly for local prediction."
        ) from exc

    uploaded = files.upload()
    if not uploaded:
        print("No file uploaded.")
        return

    file_name = list(uploaded.keys())[0]
    try:
        predict_emotion(
            model_path=model_path,
            image_path=file_name,
            class_map_path=class_map_path,
        )
    except Exception as err:
        print("Prediction failed:", err)

    print(
        "\nNote: This model predicts only the seven trained emotion classes "
        "(Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral)."
    )
    print("It is an educational model and not a medical or psychological diagnostic tool.")


def parse_args():
    parser = argparse.ArgumentParser(description="Predict facial emotion from an image.")
    parser.add_argument("--model_path", type=str, default="models/best_emotion_model.keras")
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--class_map_path", type=str, default="models/class_mapping.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predict_emotion(
        model_path=args.model_path,
        image_path=args.image_path,
        class_map_path=args.class_map_path,
    )
