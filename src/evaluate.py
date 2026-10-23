import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from data_preprocessing import EMOTION_LABELS, prepare_fer2013_data


def evaluate(
    csv_path: str,
    model_path: str = "models/best_emotion_model.keras",
    validation_size: float = 0.1,
    random_state: int = 42,
    output_dir: str = "results/metrics",
):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}. Train the model first."
        )

    os.makedirs(output_dir, exist_ok=True)

    data = prepare_fer2013_data(
        csv_path=csv_path,
        validation_size=validation_size,
        random_state=random_state,
    )
    X_test = data["X_test"]
    y_test = data["y_test"]

    model = tf.keras.models.load_model(model_path)

    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    y_prob = model.predict(X_test, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)

    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    class_names = [EMOTION_LABELS[i] for i in range(len(EMOTION_LABELS))]
    report_text = classification_report(
        y_test,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
    )

    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(class_names))))

    print("\nTest Loss:", test_loss)
    print("Test Accuracy:", test_accuracy)
    print("Weighted Precision:", precision)
    print("Weighted Recall:", recall)
    print("Weighted F1-score:", f1)
    print("\nClassification Report:\n")
    print(report_text)

    metrics_summary = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "weighted_precision": float(precision),
        "weighted_recall": float(recall),
        "weighted_f1_score": float(f1),
    }

    with open(os.path.join(output_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    with open(
        os.path.join(output_dir, "classification_report.txt"), "w", encoding="utf-8"
    ) as f:
        f.write(report_text)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.show()

    print(f"\nSaved metrics to: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained FER2013 model.")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to fer2013.csv")
    parser.add_argument("--model_path", type=str, default="models/best_emotion_model.keras")
    parser.add_argument("--validation_size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results/metrics")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        csv_path=args.csv_path,
        model_path=args.model_path,
        validation_size=args.validation_size,
        random_state=args.seed,
        output_dir=args.output_dir,
    )
