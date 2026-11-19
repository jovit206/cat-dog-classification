"""Custom CNN building blocks and deterministic input pipelines for PawVision."""
from __future__ import annotations

from collections.abc import Sequence

import tensorflow as tf

DEFAULT_IMAGE_SIZE = (128, 128)
CLASS_NAMES = ("Dog", "Cat")  # Dog=0 and Cat=1; the sigmoid estimates P(Cat).


def make_dataset(
    directory: str,
    batch_size: int = 32,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    shuffle: bool = False,
    seed: int = 42,
) -> tf.data.Dataset:
    """Read class folders, convert to float RGB [0, 1], and prefetch batches."""
    dataset = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="binary",
        class_names=list(CLASS_NAMES),
        color_mode="rgb",
        image_size=image_size,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
    )
    dataset = dataset.map(
        lambda images, labels: (tf.cast(images, tf.float32) / 255.0, labels),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )
    options = tf.data.Options()
    options.experimental_deterministic = True
    return dataset.with_options(options).prefetch(tf.data.AUTOTUNE)


def _augmentation_layers(augmentation: dict[str, float] | None) -> list[tf.keras.layers.Layer]:
    """Return mild, label-preserving augmentation layers used during training only."""
    if not augmentation:
        return []
    layers: list[tf.keras.layers.Layer] = [tf.keras.layers.RandomFlip("horizontal")]
    if augmentation.get("rotation", 0) > 0:
        layers.append(tf.keras.layers.RandomRotation(augmentation["rotation"], fill_mode="reflect"))
    if augmentation.get("zoom", 0) > 0:
        layers.append(tf.keras.layers.RandomZoom(augmentation["zoom"], fill_mode="reflect"))
    if augmentation.get("translation", 0) > 0:
        layers.append(tf.keras.layers.RandomTranslation(augmentation["translation"], augmentation["translation"], fill_mode="reflect"))
    if augmentation.get("brightness", 0) > 0:
        layers.append(tf.keras.layers.RandomBrightness(augmentation["brightness"], value_range=(0.0, 1.0)))
    if augmentation.get("contrast", 0) > 0:
        layers.append(tf.keras.layers.RandomContrast(augmentation["contrast"]))
    return layers


def build_model(
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    filters: Sequence[int] = (32, 64, 128, 192),
    dropout: float = 0.25,
    learning_rate: float = 3e-4,
    optimizer_name: str = "adam",
    weight_decay: float = 1e-4,
    augmentation: dict[str, float] | None = None,
) -> tf.keras.Model:
    """Build a scratch CNN with configurable, validation-comparable settings."""
    if not filters or any(width < 8 for width in filters):
        raise ValueError("filters must contain positive channel widths")
    if not 0 <= dropout < 0.8:
        raise ValueError("dropout must be in [0, 0.8)")
    inputs = tf.keras.Input(shape=(*image_size, 3), name="image")
    x = inputs
    for layer in _augmentation_layers(augmentation):
        x = layer(x)

    # Downsample in the stem to keep CPU training practical while retaining
    # progressively wider learned feature maps.
    stem_width = filters[0]
    x = tf.keras.layers.Conv2D(stem_width, 3, strides=2, padding="same", use_bias=False, kernel_initializer="he_normal", name="stem_conv")(x)
    x = tf.keras.layers.BatchNormalization(name="stem_bn")(x)
    x = tf.keras.layers.ReLU(name="stem_relu")(x)
    for index, width in enumerate(filters):
        x = tf.keras.layers.Conv2D(width, 3, padding="same", use_bias=False, kernel_initializer="he_normal", name=f"conv_{index + 1}")(x)
        x = tf.keras.layers.BatchNormalization(name=f"bn_{index + 1}")(x)
        x = tf.keras.layers.ReLU(name=f"relu_{index + 1}")(x)
        x = tf.keras.layers.MaxPooling2D(name=f"pool_{index + 1}")(x)
        block_dropout = min(0.35, dropout * (0.35 + index * 0.18))
        if block_dropout:
            x = tf.keras.layers.Dropout(block_dropout, name=f"dropout_{index + 1}")(x)

    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = tf.keras.layers.Dense(160, use_bias=False, name="dense_features")(x)
    x = tf.keras.layers.BatchNormalization(name="dense_batch_norm")(x)
    x = tf.keras.layers.ReLU(name="dense_relu")(x)
    x = tf.keras.layers.Dropout(dropout, name="dense_dropout")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="cat_probability")(x)
    model = tf.keras.Model(inputs, outputs, name="pawvision_custom_cnn")

    optimizer_name = optimizer_name.lower()
    if optimizer_name == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == "adamw":
        optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError("optimizer_name must be 'adam' or 'adamw'")
    model.compile(optimizer=optimizer, loss=tf.keras.losses.BinaryCrossentropy(), metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy")])
    return model
