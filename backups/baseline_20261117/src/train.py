"""Train the PawVision CNN. Metrics are written only after real training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import tensorflow as tf

from model import build_model, make_dataset

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--data", type=Path, default=ROOT / "dataset")
    args = parser.parse_args()
    if not 1 <= args.epochs <= 100:
        parser.error("--epochs must be between 1 and 100")
    train_dir, val_dir = args.data / "train", args.data / "validation"
    if not train_dir.is_dir() or not val_dir.is_dir():
        raise SystemExit("Dataset splits missing. Run src/preprocessing.py first.")
    tf.keras.utils.set_random_seed(42)
    train_ds = make_dataset(str(train_dir), args.batch_size, shuffle=True)
    val_ds = make_dataset(str(val_dir), args.batch_size)
    model_path = ROOT / "models" / "dogs_cats_model.keras"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(model_path, monitor="val_accuracy", save_best_only=True, mode="max", verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1),
    ]
    model = build_model()
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)
    (ROOT / "outputs").mkdir(exist_ok=True)
    with (ROOT / "outputs" / "training_history.json").open("w", encoding="utf-8") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=2)
    print(f"Best model saved to: {model_path}")


if __name__ == "__main__":
    main()
