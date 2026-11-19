"""Evaluate a trained PawVision CNN on the held-out test split."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score

from model import make_dataset

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    model_path, test_dir = ROOT / "models" / "dogs_cats_model.keras", ROOT / "dataset" / "test"
    if not model_path.is_file():
        raise SystemExit(f"Trained model missing: {model_path}. Run src/train.py first.")
    if not test_dir.is_dir():
        raise SystemExit(f"Test split missing: {test_dir}. Run src/preprocessing.py first.")
    model = tf.keras.models.load_model(model_path)
    ds = make_dataset(str(test_dir), shuffle=False)
    y_true = np.concatenate([labels.numpy().ravel().astype(int) for _, labels in ds])
    loss, accuracy = model.evaluate(ds, verbose=1)
    probabilities = model.predict(ds, verbose=1).ravel()
    y_pred = (probabilities >= 0.5).astype(int)
    report = classification_report(y_true, y_pred, labels=[0, 1], target_names=["Dog", "Cat"], digits=4, zero_division=0)
    metrics = {"test_loss": float(loss), "test_accuracy": float(accuracy),
               "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
               "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
               "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
               "test_images": int(len(y_true))}
    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out / "classification_report.txt").write_text(report, encoding="utf-8")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(6, 5)); sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Dog", "Cat"], yticklabels=["Dog", "Cat"])
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Test confusion matrix"); plt.tight_layout(); plt.savefig(out / "confusion_matrix.png", dpi=160); plt.close()
    history_path = out / "training_history.json"
    if history_path.exists():
        history = json.loads(history_path.read_text(encoding="utf-8"))
        for key, title, filename in (("accuracy", "Training and validation accuracy", "accuracy.png"), ("loss", "Training and validation loss", "loss.png")):
            plt.figure(figsize=(7, 4))
            plt.plot(history.get(key, []), label="Training")
            plt.plot(history.get("val_" + key, []), label="Validation")
            plt.xlabel("Epoch"); plt.ylabel(key.capitalize()); plt.title(title); plt.legend(); plt.tight_layout(); plt.savefig(out / filename, dpi=160); plt.close()
    print(json.dumps(metrics, indent=2))
    print(report)


if __name__ == "__main__":
    main()
