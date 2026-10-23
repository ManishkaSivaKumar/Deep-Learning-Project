import json
import os
import random
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

EMOTION_LABELS = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral",
}

REQUIRED_COLUMNS = ["emotion", "pixels"]
OPTIONAL_USAGE_COLUMN = "Usage"


def set_random_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_fer2013_csv(csv_path: str) -> pd.DataFrame:
    """Load FER2013 CSV and validate required columns."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Dataset file not found: {csv_path}. "
            "Upload fer2013.csv to your working directory in Colab, then retry."
        )

    df = pd.read_csv(csv_path)
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}. "
            "Expected columns include: emotion, pixels (and optionally Usage)."
        )

    return df


def inspect_dataset(df: pd.DataFrame) -> None:
    """Print quick dataset summary for beginners."""
    print("Dataset shape:", df.shape)
    print("Columns:", list(df.columns))
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nMissing values by column:")
    print(df.isna().sum())


def _parse_pixel_string(pixel_string: str) -> np.ndarray:
    """Convert one FER2013 pixel string to a 48x48 float32 array."""
    values = pixel_string.strip().split()
    if len(values) != 48 * 48:
        raise ValueError("Pixel count is not 2304")

    arr = np.array(values, dtype=np.float32)
    if np.any(arr < 0) or np.any(arr > 255):
        raise ValueError("Pixel values must be in 0 to 255")

    return arr.reshape((48, 48, 1))


def clean_and_convert(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, Dict[str, int]]:
    """
    Validate and convert FER rows to image tensors.

    Returns:
    - X: image array of shape (N, 48, 48, 1), normalized to [0, 1]
    - y: label array of shape (N,)
    - clean_df: dataframe with only valid rows
    - dropped_info: simple counters for dropped rows
    """
    dropped_info = {
        "missing_emotion_or_pixels": 0,
        "invalid_label": 0,
        "invalid_pixels": 0,
    }

    valid_indices = []
    images = []
    labels = []

    for idx, row in df.iterrows():
        emotion = row.get("emotion")
        pixels = row.get("pixels")

        if pd.isna(emotion) or pd.isna(pixels):
            dropped_info["missing_emotion_or_pixels"] += 1
            continue

        try:
            emotion_int = int(emotion)
        except Exception:
            dropped_info["invalid_label"] += 1
            continue

        if emotion_int not in EMOTION_LABELS:
            dropped_info["invalid_label"] += 1
            continue

        try:
            img = _parse_pixel_string(str(pixels))
        except Exception:
            dropped_info["invalid_pixels"] += 1
            continue

        valid_indices.append(idx)
        images.append(img)
        labels.append(emotion_int)

    if not images:
        raise ValueError(
            "No valid rows found after validation. "
            "Check whether the CSV matches FER2013 format."
        )

    clean_df = df.loc[valid_indices].reset_index(drop=True)
    X = np.array(images, dtype=np.float32) / 255.0
    y = np.array(labels, dtype=np.int64)

    return X, y, clean_df, dropped_info


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    clean_df: pd.DataFrame,
    validation_size: float = 0.1,
    random_state: int = 42,
):
    """
    Split dataset into train/val/test with no leakage.

    If Usage column exists:
    - Train split: Usage == Training
    - Test split: Usage == PublicTest (preferred). If absent, use PrivateTest.
    - Validation split: stratified split from training split only.

    If Usage column does not exist:
    - Stratified split into train+val and test, then train and val.
    """
    usage_available = OPTIONAL_USAGE_COLUMN in clean_df.columns

    if usage_available:
        usage_values = clean_df[OPTIONAL_USAGE_COLUMN].astype(str).str.strip().str.lower()

        train_mask = usage_values == "training"
        public_test_mask = usage_values == "publictest"
        private_test_mask = usage_values == "privatetest"

        if public_test_mask.sum() > 0:
            test_mask = public_test_mask
            test_partition_name = "PublicTest"
        elif private_test_mask.sum() > 0:
            test_mask = private_test_mask
            test_partition_name = "PrivateTest"
        else:
            raise ValueError(
                "Usage column exists, but no PublicTest or PrivateTest rows were found."
            )

        if train_mask.sum() == 0:
            raise ValueError("Usage column exists, but no Training rows were found.")

        X_train_all = X[train_mask.values]
        y_train_all = y[train_mask.values]
        X_test = X[test_mask.values]
        y_test = y[test_mask.values]

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_all,
            y_train_all,
            test_size=validation_size,
            random_state=random_state,
            stratify=y_train_all,
        )

        split_info = {
            "used_predefined_partitions": True,
            "test_partition": test_partition_name,
        }
    else:
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X,
            y,
            test_size=0.15,
            random_state=random_state,
            stratify=y,
        )

        val_ratio_within_train = validation_size / (1.0 - 0.15)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val,
            y_train_val,
            test_size=val_ratio_within_train,
            random_state=random_state,
            stratify=y_train_val,
        )

        split_info = {
            "used_predefined_partitions": False,
            "test_partition": "Stratified split",
        }

    return X_train, y_train, X_val, y_val, X_test, y_test, split_info


def print_class_distribution(y: np.ndarray, split_name: str = "dataset") -> None:
    """Print class counts for one split."""
    print(f"\nClass distribution in {split_name}:")
    unique, counts = np.unique(y, return_counts=True)
    for class_id, count in zip(unique, counts):
        print(f"  {class_id} ({EMOTION_LABELS.get(int(class_id), 'Unknown')}): {int(count)}")


def plot_class_distribution(y: np.ndarray, title: str = "Class Distribution") -> None:
    """Plot class counts using matplotlib only."""
    unique, counts = np.unique(y, return_counts=True)
    labels = [EMOTION_LABELS.get(int(idx), str(idx)) for idx in unique]

    plt.figure(figsize=(9, 4))
    plt.bar(labels, counts)
    plt.title(title)
    plt.xlabel("Emotion")
    plt.ylabel("Count")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.show()


def show_sample_images(X: np.ndarray, y: np.ndarray, sample_count: int = 12) -> None:
    """Display sample images with labels."""
    sample_count = min(sample_count, len(X))
    indices = np.random.choice(len(X), size=sample_count, replace=False)

    cols = 4
    rows = int(np.ceil(sample_count / cols))

    plt.figure(figsize=(12, 3 * rows))
    for i, idx in enumerate(indices, start=1):
        plt.subplot(rows, cols, i)
        plt.imshow(X[idx].reshape(48, 48), cmap="gray")
        plt.title(EMOTION_LABELS.get(int(y[idx]), "Unknown"))
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def save_class_mapping(path: str) -> None:
    """Save class id to emotion label mapping as JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(EMOTION_LABELS, f, indent=2)


def prepare_fer2013_data(
    csv_path: str,
    validation_size: float = 0.1,
    random_state: int = 42,
):
    """Full data preparation pipeline used by training/evaluation scripts."""
    set_random_seeds(random_state)
    df = load_fer2013_csv(csv_path)
    inspect_dataset(df)

    X, y, clean_df, dropped_info = clean_and_convert(df)
    print("\nDropped rows summary:", dropped_info)

    X_train, y_train, X_val, y_val, X_test, y_test, split_info = split_data(
        X,
        y,
        clean_df,
        validation_size=validation_size,
        random_state=random_state,
    )

    print("\nSplit info:", split_info)
    print("Train shape:", X_train.shape, y_train.shape)
    print("Validation shape:", X_val.shape, y_val.shape)
    print("Test shape:", X_test.shape, y_test.shape)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "split_info": split_info,
        "dropped_info": dropped_info,
        "clean_row_count": len(clean_df),
    }
