import argparse
import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from data_preprocessing import (
    EMOTION_LABELS,
    plot_class_distribution,
    prepare_fer2013_data,
    save_class_mapping,
    show_sample_images,
)
from model import build_cnn_model


def set_seeds(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def plot_training_history(history, output_dir: str) -> None:
    """Save training and validation curves."""
    os.makedirs(output_dir, exist_ok=True)

    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])

    plt.figure(figsize=(8, 5))
    plt.plot(acc, label="Training Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_curve.png"))
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_curve.png"))
    plt.show()


def train(
    csv_path: str,
    model_output_path: str = "models/best_emotion_model.keras",
    history_output_path: str = "results/history.json",
    plots_dir: str = "results/plots",
    epochs: int = 20,
    batch_size: int = 64,
    validation_size: float = 0.1,
    random_state: int = 42,
    use_augmentation: bool = True,
):
    """Complete training pipeline."""
    set_seeds(random_state)

    # Try to use GPU if available, but continue on CPU if not.
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print("GPU detected. Training will use GPU acceleration.")
    else:
        print("No GPU detected. Training will run on CPU.")

    data = prepare_fer2013_data(
        csv_path=csv_path,
        validation_size=validation_size,
        random_state=random_state,
    )

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]

    print("\nShowing class distribution and sample training images...")
    plot_class_distribution(y_train, title="Training Class Distribution")
    show_sample_images(X_train, y_train, sample_count=12)

    model = build_cnn_model(input_shape=(48, 48, 1), num_classes=len(EMOTION_LABELS))
    model.summary()

    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(history_output_path), exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=model_output_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    if use_augmentation:
        print("\nUsing data augmentation for training set only.")
        train_datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
        )
        train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)

        history = model.fit(
            train_generator,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1,
        )
    else:
        print("\nTraining without augmentation.")
        history = model.fit(
            X_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1,
        )

    with open(history_output_path, "w", encoding="utf-8") as f:
        json.dump(history.history, f, indent=2)

    plot_training_history(history, plots_dir)

    class_map_path = os.path.join(os.path.dirname(model_output_path), "class_mapping.json")
    save_class_mapping(class_map_path)

    print("\nTraining complete.")
    print(f"Best model saved to: {model_output_path}")
    print(f"Class mapping saved to: {class_map_path}")
    print(f"Training history saved to: {history_output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train CNN on FER2013 dataset.")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to fer2013.csv")
    parser.add_argument("--model_output", type=str, default="models/best_emotion_model.keras")
    parser.add_argument("--history_output", type=str, default="results/history.json")
    parser.add_argument("--plots_dir", type=str, default="results/plots")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--validation_size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_augmentation", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        csv_path=args.csv_path,
        model_output_path=args.model_output,
        history_output_path=args.history_output,
        plots_dir=args.plots_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_size=args.validation_size,
        random_state=args.seed,
        use_augmentation=not args.no_augmentation,
    )
