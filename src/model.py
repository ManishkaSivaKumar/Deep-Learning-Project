import tensorflow as tf
from tensorflow.keras import layers, models


def build_cnn_model(input_shape=(48, 48, 1), num_classes=7):
    """
    Build a simple CNN model for FER2013 emotion classification.

    Architecture:
    1) Conv(32) + ReLU
    2) MaxPool
    3) Conv(64) + ReLU
    4) MaxPool
    5) Conv(128) + ReLU
    6) MaxPool
    7) Flatten
    8) Dense(128) + ReLU
    9) Dropout(0.5)
    10) Dense(num_classes) + Softmax
    """
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
