"""Evaluate the validation-selected candidate once on the untouched test split."""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from model import CLASS_NAMES, make_dataset
from paths import BASELINE_BACKUP, CANDIDATE_METADATA, CANDIDATE_MODEL, DATASET_128, OUTPUTS, PRODUCTION_METADATA, PRODUCTION_MODEL, ROOT, SCRATCH_BACKUP

DATASET = DATASET_128  # Overridden by --dataset or the candidate metadata's "dataset" entry.


def _plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Final test confusion matrix")
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)


def _plot_curves(y_true: np.ndarray, p_cat: np.ndarray) -> tuple[float, float]:
    fpr, tpr, _ = roc_curve(y_true, p_cat)
    auc = float(roc_auc_score(y_true, p_cat))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"ROC AUC = {auc:.4f}", color="#4169d8", linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#95a0b2", label="Chance")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="Final test ROC curve", xlim=(0, 1), ylim=(0, 1.01))
    ax.legend(loc="lower right"); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(OUTPUTS / "roc_curve.png", dpi=170); plt.close(fig)

    precision, recall, _ = precision_recall_curve(y_true, p_cat)
    average_precision = float(average_precision_score(y_true, p_cat))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color="#249b88", linewidth=2, label=f"Average precision = {average_precision:.4f}")
    ax.set(xlabel="Recall (Cat is positive class)", ylabel="Precision", title="Final test precision-recall curve", xlim=(0, 1), ylim=(0, 1.01))
    ax.legend(loc="lower left"); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(OUTPUTS / "precision_recall_curve.png", dpi=170); plt.close(fig)
    return auc, average_precision


def _plot_history(history_path: Path) -> None:
    if not history_path.is_file():
        return
    history = json.loads(history_path.read_text(encoding="utf-8"))
    for key, title, filename in (("accuracy", "Training and validation accuracy", "accuracy.png"), ("loss", "Training and validation loss", "loss.png")):
        if key not in history or f"val_{key}" not in history:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(history[key], label="Training", color="#456be2")
        ax.plot(history[f"val_{key}"], label="Validation", color="#35a494")
        ax.set_xlabel("Epoch"); ax.set_ylabel(key.capitalize()); ax.set_title(title); ax.legend(); ax.grid(alpha=.18)
        fig.tight_layout(); fig.savefig(OUTPUTS / filename, dpi=160); plt.close(fig)


def _save_test_error_examples(y_true: np.ndarray, y_pred: np.ndarray, p_cat: np.ndarray, threshold: float = 0.5) -> None:
    """Create examples only after final scoring; these are not fed back into selection."""
    out_dir = OUTPUTS / "error_analysis" / "final_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [path for label in CLASS_NAMES for path in sorted((DATASET / "test" / label).glob("*.jpg"))]
    if len(paths) != len(y_true):
        raise RuntimeError(f"Test path/label count mismatch: {len(paths)} != {len(y_true)}")
    groups = {
        "false_negative_cats": np.where((y_true == 1) & (y_pred == 0))[0],
        "false_positive_cats": np.where((y_true == 0) & (y_pred == 1))[0],
        "correct_dogs": np.where((y_true == 0) & (y_pred == 0))[0],
        "correct_cats": np.where((y_true == 1) & (y_pred == 1))[0],
    }
    for name, indices in groups.items():
        indices = sorted(indices, key=lambda i: abs(float(p_cat[i]) - threshold), reverse=True)[:12]
        columns, rows = 4, max(1, (len(indices) + 3) // 4)
        fig, axes = plt.subplots(rows, columns, figsize=(12, rows * 3))
        axes = np.asarray(axes).reshape(-1)
        for ax, index in zip(axes, indices):
            with Image.open(paths[index]) as image:
                ax.imshow(image.convert("RGB"))
            ax.axis("off")
            ax.set_title(f"True {CLASS_NAMES[y_true[index]]} · Pred {CLASS_NAMES[y_pred[index]]}\nP(Cat)={p_cat[index]:.2f}", fontsize=9)
        for ax in axes[len(indices):]:
            ax.axis("off")
        fig.suptitle(f"Final test examples · {name.replace('_', ' ')} ({len(indices)} shown)")
        fig.tight_layout(); fig.savefig(out_dir / f"{name}.png", dpi=130); plt.close(fig)


BASELINES = (
    ("Original baseline", BASELINE_BACKUP / "outputs"),
    ("Scratch CNN", SCRATCH_BACKUP / "outputs"),
)


def _class_rows(report_path: Path) -> dict[str, list[float]]:
    rows: dict[str, list[float]] = {}
    if report_path.is_file():
        for line in report_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) >= 4 and parts[0] in CLASS_NAMES:
                rows[parts[0]] = [float(value) for value in parts[1:4]]
    return rows


def _comparison_report(metrics: dict[str, Any], threshold: float, experiment: str) -> None:
    columns: list[tuple[str, dict[str, Any], dict[str, list[float]]]] = []
    for label, folder in BASELINES:
        metrics_path = folder / "metrics.json"
        if metrics_path.is_file():
            columns.append((label, json.loads(metrics_path.read_text(encoding="utf-8")), _class_rows(folder / "classification_report.txt")))
    columns.append(("Selected model", metrics, _class_rows(OUTPUTS / "classification_report.txt")))
    header = "| Metric | " + " | ".join(label for label, _, _ in columns) + " |"
    lines = [
        "# PawVision final model comparison",
        "",
        f"Selected using validation macro F1: **{experiment}**. The validation-selected Cat threshold was **{threshold:.3f}**. "
        "The test split was not used for model or threshold selection.",
        "",
        header,
        "|---|" + "---:|" * len(columns),
    ]
    for title, key, percentage in (
        ("Test accuracy", "test_accuracy", True),
        ("Test loss", "test_loss", False),
        ("Macro precision", "precision_macro", True),
        ("Macro recall", "recall_macro", True),
        ("Macro F1", "f1_macro", True),
        ("ROC AUC", "roc_auc", False),
    ):
        cells = []
        for _, values, _ in columns:
            value = values.get(key)
            cells.append("—" if not isinstance(value, (int, float)) else (f"{value*100:.2f}%" if percentage else f"{value:.4f}"))
        lines.append(f"| {title} | " + " | ".join(cells) + " |")
    lines += ["", "## Per-class test F1", "", header, "|---|" + "---:|" * len(columns)]
    for label in CLASS_NAMES:
        cells = [f"{rows[label][2]*100:.2f}%" if label in rows else "—" for _, _, rows in columns]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "The original baseline used threshold 0.5; later models use a threshold chosen on validation data only.",
    ]
    (OUTPUTS / "final_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=CANDIDATE_MODEL, help="Validation-selected candidate model")
    parser.add_argument("--metadata", type=Path, default=CANDIDATE_METADATA)
    parser.add_argument("--dataset", type=Path, default=None, help="Dataset root with a test/ split (defaults to the metadata's dataset, else data/processed_128)")
    parser.add_argument("--promote", action="store_true", help="Promote the already validation-selected candidate after its one final test evaluation")
    parser.add_argument("--allow-repeat-test", action="store_true", help="Explicitly allow re-scoring the final test split; avoid for the primary result")
    args = parser.parse_args()
    model_path = args.model if args.model.is_absolute() else ROOT / args.model
    metadata_path = args.metadata if args.metadata.is_absolute() else ROOT / args.metadata
    result_guard = OUTPUTS / "final_metrics.json"
    if result_guard.exists() and not args.allow_repeat_test:
        raise SystemExit(f"Final test evaluation already exists at {result_guard}. Refusing to reuse the holdout; pass --allow-repeat-test only for an explicit repeat.")
    if not model_path.is_file():
        raise SystemExit(f"Candidate model not found: {model_path}. Run src/train_transfer.py (or src/train_scratch.py) first.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    global DATASET
    if args.dataset is not None:
        DATASET = args.dataset if args.dataset.is_absolute() else ROOT / args.dataset
    elif metadata.get("dataset"):
        DATASET = ROOT / metadata["dataset"]
    test_dir = DATASET / "test"
    if not test_dir.is_dir():
        raise SystemExit(f"Test split not found: {test_dir}")
    threshold = float(metadata.get("validation_threshold", 0.5))
    model = tf.keras.models.load_model(model_path)
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]
    image_size = (int(input_shape[1]), int(input_shape[2]))
    batch_size = int(metadata.get("batch_size", 32))
    test_ds = make_dataset(str(test_dir), batch_size=batch_size, image_size=image_size, shuffle=False)

    # Read, label, and score each test batch in a single pass.
    labels_by_batch: list[np.ndarray] = []
    probabilities_by_batch: list[np.ndarray] = []
    for images, labels in test_ds:
        probabilities_by_batch.append(model(images, training=False).numpy().reshape(-1).astype(np.float64))
        labels_by_batch.append(labels.numpy().reshape(-1).astype(np.int32))
    y_true = np.concatenate(labels_by_batch)
    p_cat = np.concatenate(probabilities_by_batch)
    y_pred = (p_cat >= threshold).astype(np.int32)
    bce = tf.keras.losses.binary_crossentropy(y_true.astype(np.float32), p_cat.astype(np.float32)).numpy()
    metrics: dict[str, Any] = {
        "test_loss": float(np.mean(bce)),
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, p_cat)),
        "average_precision": float(average_precision_score(y_true, p_cat)),
        "test_images": int(len(y_true)),
        "decision_threshold": threshold,
        "positive_class": "Cat",
        "image_size": list(image_size),
    }
    report = classification_report(y_true, y_pred, labels=[0, 1], target_names=list(CLASS_NAMES), digits=4, zero_division=0)
    metrics["dog_precision"] = float(precision_score(y_true, y_pred, labels=[0], average="macro", zero_division=0))
    metrics["dog_recall"] = float(recall_score(y_true, y_pred, labels=[0], average="macro", zero_division=0))
    metrics["cat_precision"] = float(precision_score(y_true, y_pred, labels=[1], average="macro", zero_division=0))
    metrics["cat_recall"] = float(recall_score(y_true, y_pred, labels=[1], average="macro", zero_division=0))

    OUTPUTS.mkdir(exist_ok=True)
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    _plot_confusion(y_true, y_pred, OUTPUTS / "confusion_matrix.png")
    auc, average_precision = _plot_curves(y_true, p_cat)
    metrics["roc_auc"], metrics["average_precision"] = auc, average_precision
    (OUTPUTS / "classification_report.txt").write_text(report, encoding="utf-8")
    (OUTPUTS / "final_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (OUTPUTS / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _save_test_error_examples(y_true, y_pred, p_cat, threshold)

    experiment = str(metadata.get("selected_experiment", model_path.stem))
    _comparison_report(metrics, threshold, experiment)

    if args.promote:
        # Selection is already fixed from validation. Promotion does not depend on test metrics.
        production = PRODUCTION_MODEL
        temporary = PRODUCTION_MODEL.with_name(".dogs_cats_model.promoting.keras")
        shutil.copy2(model_path, temporary)
        os.replace(temporary, production)
        promoted_metadata = {
            **metadata,
            "production_model": "models/dogs_cats_model.keras",
            "promoted_from": str(model_path.relative_to(ROOT).as_posix()) if model_path.is_relative_to(ROOT) else str(model_path),
            "final_test_evaluated_once": True,
            "final_test_metrics": {key: metrics[key] for key in ("test_accuracy", "test_loss", "precision_macro", "recall_macro", "f1_macro", "f1_weighted", "roc_auc", "average_precision")},
        }
        PRODUCTION_METADATA.write_text(json.dumps(promoted_metadata, indent=2), encoding="utf-8")
        history = OUTPUTS / "experiments" / experiment / "history.json"
        if history.is_file():
            shutil.copy2(history, OUTPUTS / "training_history.json")
            _plot_history(OUTPUTS / "training_history.json")
        print(f"Promoted validation-selected model to {production}")

    print(json.dumps({key: metrics[key] for key in ("test_accuracy", "test_loss", "precision_macro", "recall_macro", "f1_macro", "f1_weighted", "roc_auc", "average_precision", "decision_threshold")}, indent=2))
    print(report)
    print(f"Final metrics and charts saved under {OUTPUTS}")


if __name__ == "__main__":
    main()
