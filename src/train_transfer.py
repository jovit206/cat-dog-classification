"""Fine-tune ImageNet-pretrained backbones on the PawVision split and stage the validation winner.

Two-stage recipe per backbone:
  1. Freeze the pretrained backbone and train only the new classification head.
  2. Unfreeze the top of the backbone and fine-tune with a small learning rate.

Every model takes float RGB in [0, 1] (the same contract as the scratch CNN and the app)
and does its own backbone-specific preprocessing internally. The sigmoid output is P(Cat).
Selection uses validation macro F1 only; the test split is never read here.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from model import CLASS_NAMES, make_dataset
from train import choose_threshold

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset_224"
OUTPUTS = ROOT / "outputs"
MODELS = ROOT / "models"
IMAGE_SIZE = (224, 224)

BACKBONES: dict[str, dict[str, Any]] = {
    "mobilenetv2_224_finetune": {
        "builder": lambda: tf.keras.applications.MobileNetV2(input_shape=(*IMAGE_SIZE, 3), include_top=False, weights="imagenet"),
        # MobileNetV2 expects [-1, 1].
        "preprocess": lambda: tf.keras.layers.Rescaling(2.0, offset=-1.0, name="preprocess"),
        "fine_tune_layers": 40,
    },
    "efficientnetv2b0_224_finetune": {
        "builder": lambda: tf.keras.applications.EfficientNetV2B0(input_shape=(*IMAGE_SIZE, 3), include_top=False, weights="imagenet", include_preprocessing=True),
        # EfficientNetV2 includes its own normalization and expects [0, 255].
        "preprocess": lambda: tf.keras.layers.Rescaling(255.0, name="preprocess"),
        "fine_tune_layers": 60,
    },
}


def build_transfer_model(name: str) -> tuple[tf.keras.Model, tf.keras.Model]:
    spec = BACKBONES[name]
    backbone = spec["builder"]()
    backbone.name = "backbone"  # The app looks up this layer to draw its attention heatmap.
    backbone.trainable = False
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3), name="image")
    x = tf.keras.layers.RandomFlip("horizontal", name="aug_flip")(inputs)
    x = tf.keras.layers.RandomRotation(0.05, fill_mode="reflect", name="aug_rotate")(x)
    x = tf.keras.layers.RandomZoom(0.10, fill_mode="reflect", name="aug_zoom")(x)
    x = tf.keras.layers.RandomContrast(0.10, name="aug_contrast")(x)
    x = spec["preprocess"]()(x)
    # training=False keeps BatchNorm statistics frozen even while fine-tuning.
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="pool")(x)
    x = tf.keras.layers.Dropout(0.25, name="head_dropout")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="cat_probability")(x)
    return tf.keras.Model(inputs, outputs, name=f"pawvision_{name}"), backbone


def _compile(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy")],
    )


def score(model: tf.keras.Model, dataset: tf.data.Dataset) -> dict[str, Any]:
    y_true = np.concatenate([labels.numpy().reshape(-1).astype(np.int32) for _, labels in dataset])
    p_cat = model.predict(dataset, verbose=0).reshape(-1).astype(np.float64)
    threshold, metrics = choose_threshold(y_true, p_cat)
    y_pred = (p_cat >= threshold).astype(np.int32)
    report = classification_report(y_true, y_pred, labels=[0, 1], target_names=list(CLASS_NAMES), output_dict=True, zero_division=0)
    bce = float(np.mean(tf.keras.losses.binary_crossentropy(y_true.astype(np.float32)[:, None], p_cat.astype(np.float32)[:, None]).numpy()))
    return {
        **metrics,
        "threshold": float(threshold),
        "loss": bce,
        "accuracy_at_0_5": float(np.mean((p_cat >= 0.5).astype(np.int32) == y_true)),
        "dog_recall": float(report["Dog"]["recall"]),
        "cat_recall": float(report["Cat"]["recall"]),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "y_true": y_true,
        "p_cat": p_cat,
    }


def run(name: str, head_epochs: int, fine_tune_epochs: int, patience: int, seed: int, batch_size: int) -> dict[str, Any]:
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    train_ds = make_dataset(str(DATASET / "train"), batch_size, IMAGE_SIZE, shuffle=True, seed=seed)
    val_ds = make_dataset(str(DATASET / "validation"), batch_size, IMAGE_SIZE, shuffle=False, seed=seed)
    out_dir = OUTPUTS / "experiments" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / "best_model.keras"
    model, backbone = build_transfer_model(name)

    def callbacks(stage: str) -> list[tf.keras.callbacks.Callback]:
        return [
            tf.keras.callbacks.ModelCheckpoint(checkpoint, monitor="val_loss", mode="min", save_best_only=True, verbose=0),
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=patience, restore_best_weights=True, verbose=1),
            tf.keras.callbacks.CSVLogger(out_dir / f"training_log_{stage}.csv"),
        ]

    start = time.perf_counter()
    print(f"[{name}] stage 1: training the new head with the backbone frozen")
    _compile(model, 1e-3)
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=head_epochs, callbacks=callbacks("head"), verbose=2)

    print(f"[{name}] stage 2: fine-tuning the top {BACKBONES[name]['fine_tune_layers']} backbone layers")
    backbone.trainable = True
    cutoff = len(backbone.layers) - BACKBONES[name]["fine_tune_layers"]
    for index, layer in enumerate(backbone.layers):
        layer.trainable = index >= cutoff and not isinstance(layer, tf.keras.layers.BatchNormalization)
    _compile(model, 2e-5)
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=fine_tune_epochs, callbacks=callbacks("finetune"), verbose=2)
    elapsed = time.perf_counter() - start

    history = {key: [float(v) for v in h1.history[key]] + [float(v) for v in h2.history.get(key, [])] for key in h1.history}
    history["fine_tune_start_epoch"] = len(h1.history["loss"])
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    model = tf.keras.models.load_model(checkpoint)
    scores = score(model, val_ds)
    candidate_path = MODELS / "experiment_candidates" / f"{name}.keras"
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint, candidate_path)
    (out_dir / "validation_metrics.json").write_text(json.dumps({k: v for k, v in scores.items() if k not in ("y_true", "p_cat")}, indent=2), encoding="utf-8")

    for key, title in (("accuracy", "Accuracy"), ("loss", "Loss")):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(history[key], label="Training")
        ax.plot(history[f"val_{key}"], label="Validation")
        ax.axvline(history["fine_tune_start_epoch"] - 0.5, color="#95a0b2", linestyle="--", label="Fine-tuning starts")
        ax.set(title=f"{name} · {title}", xlabel="Epoch", ylabel=title); ax.legend(); ax.grid(alpha=.18); fig.tight_layout()
        fig.savefig(out_dir / f"{key}.png", dpi=140); plt.close(fig)

    row = {
        "experiment": name,
        "model_path": candidate_path.relative_to(ROOT).as_posix(),
        "image_size": list(IMAGE_SIZE),
        "backbone": name.split("_")[0],
        "fine_tune_layers": BACKBONES[name]["fine_tune_layers"],
        "batch_size": batch_size,
        "epochs_run": len(history["loss"]),
        "best_val_loss": float(min(history["val_loss"])),
        "best_val_accuracy": float(max(history["val_accuracy"])),
        "validation_threshold": scores["threshold"],
        "val_accuracy": scores["accuracy"],
        "val_accuracy_at_0_5": scores["accuracy_at_0_5"],
        "val_precision_macro": scores["precision_macro"],
        "val_recall_macro": scores["recall_macro"],
        "val_f1_macro": scores["f1_macro"],
        "val_f1_weighted": scores["f1_weighted"],
        "val_dog_recall": scores["dog_recall"],
        "val_cat_recall": scores["cat_recall"],
        "val_loss": scores["loss"],
        "training_seconds": round(elapsed, 1),
        "parameters": int(model.count_params()),
    }
    print(f"[{name}] val macro F1={row['val_f1_macro']:.4f} acc={row['val_accuracy']:.4f} loss={row['val_loss']:.4f} in {elapsed/60:.1f} min")
    return row


def save_pet_gate() -> None:
    """Save the full ImageNet MobileNetV2 so the app can warn about non-pet photos offline."""
    path = MODELS / "pet_gate_imagenet.keras"
    if path.is_file():
        return
    gate = tf.keras.applications.MobileNetV2(weights="imagenet", include_top=True)
    gate.save(path)
    print(f"Saved ImageNet pet gate to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--head-epochs", type=int, default=5)
    parser.add_argument("--fine-tune-epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiments", nargs="*", choices=list(BACKBONES), default=list(BACKBONES))
    args = parser.parse_args()
    if not (DATASET / "train").is_dir():
        raise SystemExit("dataset_224 is missing. Run: python src/preprocessing.py --image-size 224 --destination dataset_224")

    save_pet_gate()
    rows = [run(name, args.head_epochs, args.fine_tune_epochs, args.patience, args.seed, args.batch_size) for name in args.experiments]
    with (OUTPUTS / "transfer_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    winner = max(rows, key=lambda r: (r["val_f1_macro"], r["val_accuracy"], -r["val_loss"]))
    candidate = MODELS / "candidate_dogs_cats_model.keras"
    shutil.copy2(ROOT / winner["model_path"], candidate)
    metadata = {
        "selected_experiment": winner["experiment"],
        "model_family": "transfer_learning",
        "backbone": winner["backbone"],
        "pretrained_on": "ImageNet",
        "candidate_model": candidate.relative_to(ROOT).as_posix(),
        "production_model": "models/dogs_cats_model.keras",
        "dataset": "dataset_224",
        "class_mapping": {"Dog": 0, "Cat": 1},
        "output": "P(Cat)",
        "input_range": [0.0, 1.0],
        "image_size": winner["image_size"],
        "batch_size": winner["batch_size"],
        "fine_tune_layers": winner["fine_tune_layers"],
        "validation_threshold": float(winner["validation_threshold"]),
        "selection_metric": "validation macro F1, then validation accuracy, then validation loss",
        "validation_metrics": {k: winner[k] for k in ("val_accuracy", "val_precision_macro", "val_recall_macro", "val_f1_macro", "val_f1_weighted", "val_dog_recall", "val_cat_recall", "val_loss")},
        "candidates": rows,
        "test_split_used_for_selection": False,
        "seed": args.seed,
    }
    (MODELS / "candidate_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Selected {winner['experiment']} on validation only (macro F1={winner['val_f1_macro']:.4f}). Candidate staged at {candidate}.")


if __name__ == "__main__":
    main()
