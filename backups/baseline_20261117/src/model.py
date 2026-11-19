"""Original CNN architecture and input pipeline for PawVision."""
from __future__ import annotations

import tensorflow as tf

IMAGE_SIZE = (128, 128)
CLASS_NAMES = ("Dog", "Cat")  # Keras directory loader sorts Cat=0, Dog=1; training overrides below.


def make_dataset(directory: str, batch_size: int = 32, shuffle: bool = False, seed: int = 42):
    """Read RGB images from Cat/Dog folders and enforce Dog=0, Cat=1 labels."""
    ds = tf.keras.utils.image_dataset_from_directory(
        directory, labels="inferred", label_mode="binary", class_names=["Dog", "Cat"],
        image_size=IMAGE_SIZE, batch_size=batch_size, shuffle=shuffle, seed=seed,
    )
    return ds.map(lambda images, labels: (tf.cast(images, tf.float32) / 255.0, labels),
                  num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)


def build_model() -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(128, 128, 3), name="image")
    x = tf.keras.layers.RandomFlip("horizontal")(inputs)
    x = tf.keras.layers.RandomRotation(0.08)(x)
    x = tf.keras.layers.RandomZoom(0.10)(x)
    x = tf.keras.layers.RandomTranslation(0.08, 0.08)(x)
    for filters in (32, 64, 128):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Dropout(0.20 if filters < 128 else 0.30)(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.40)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="cat_probability")(x)
    model = tf.keras.Model(inputs, outputs, name="pawvision_cnn")
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss="binary_crossentropy", metrics=["accuracy"])
    return model
