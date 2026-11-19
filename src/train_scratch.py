"""Run reproducible, validation-only CNN experiments and stage the best candidate."""
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
from PIL import Image
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, confusion_matrix

from model import CLASS_NAMES, DEFAULT_IMAGE_SIZE, build_model, make_dataset

from paths import BASELINE_BACKUP, CANDIDATE_METADATA, CANDIDATE_MODEL, DATASET_128, EXPERIMENT_CANDIDATES, OUTPUTS, ROOT

DATASET = DATASET_128
EXPERIMENTS: list[dict[str, Any]] = [
    {
        "name": "cnn128_adam_mild_aug",
        "image_size": (128, 128), "filters": (32, 64, 128, 192), "optimizer": "adam",
        "learning_rate": 3e-4, "weight_decay": 0.0, "batch_size": 32, "dropout": 0.25,
        "augmentation": {"rotation": 0.04, "zoom": 0.07, "translation": 0.04, "brightness": 0.08, "contrast": 0.10},
    },
    {
        "name": "cnn128_adamw_mild_aug",
        "image_size": (128, 128), "filters": (32, 64, 128, 192), "optimizer": "adamw",
        "learning_rate": 3e-4, "weight_decay": 1e-4, "batch_size": 32, "dropout": 0.25,
        "augmentation": {"rotation": 0.04, "zoom": 0.07, "translation": 0.04, "brightness": 0.08, "contrast": 0.10},
    },
    {
        "name": "cnn160_adamw_mild_aug",
        "image_size": (160, 160), "filters": (32, 64, 128, 192), "optimizer": "adamw",
        "learning_rate": 3e-4, "weight_decay": 1e-4, "batch_size": 32, "dropout": 0.25,
        "augmentation": {"rotation": 0.04, "zoom": 0.07, "translation": 0.04, "brightness": 0.08, "contrast": 0.10},
    },
    {
        "name": "cnn128_adamw_wider_dropout",
        "image_size": (128, 128), "filters": (32, 64, 128, 256), "optimizer": "adamw",
        "learning_rate": 5e-4, "weight_decay": 1e-4, "batch_size": 24, "dropout": 0.30,
        "augmentation": {"rotation": 0.04, "zoom": 0.07, "translation": 0.04, "brightness": 0.08, "contrast": 0.10},
    },
]


def choose_threshold(y_true: np.ndarray, p_cat: np.ndarray) -> tuple[float, dict[str, float]]:
    """Choose the operating threshold on validation macro F1 (never on test data)."""
    candidates = sorted(set(np.round(np.arange(0.20, 0.801, 0.01), 3).tolist() + [0.5]))
    best: tuple[tuple[float, float, float, float], float, dict[str, float]] | None = None
    for threshold in candidates:
        y_pred = (p_cat >= threshold).astype(np.int32)
        metrics = {
            "accuracy": float(np.mean(y_pred == y_true)),
            "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        }
        # Stable tie breaking prefers balanced accuracy/accuracy and a threshold near 0.5.
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        per_class_recall = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
        balanced_accuracy = float(np.mean(per_class_recall))
        key = (metrics["f1_macro"], balanced_accuracy, metrics["accuracy"], -abs(threshold - 0.5))
        if best is None or key > best[0]:
            best = (key, threshold, metrics)
    assert best is not None
    return best[1], best[2]


def score_model(model: tf.keras.Model, dataset: tf.data.Dataset) -> dict[str, Any]:
    y_true = np.concatenate([labels.numpy().reshape(-1).astype(np.int32) for _, labels in dataset])
    p_cat = model.predict(dataset, verbose=0).reshape(-1).astype(np.float64)
    threshold, selected_metrics = choose_threshold(y_true, p_cat)
    y_pred = (p_cat >= threshold).astype(np.int32)
    y_pred_05 = (p_cat >= 0.5).astype(np.int32)
    report = classification_report(y_true, y_pred, labels=[0, 1], target_names=list(CLASS_NAMES), output_dict=True, zero_division=0)
    return {
        **selected_metrics,
        "threshold": float(threshold),
        "accuracy_at_0_5": float(np.mean(y_pred_05 == y_true)),
        "dog_precision": float(report["Dog"]["precision"]),
        "dog_recall": float(report["Dog"]["recall"]),
        "cat_precision": float(report["Cat"]["precision"]),
        "cat_recall": float(report["Cat"]["recall"]),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "y_true": y_true,
        "p_cat": p_cat,
    }


def write_comparison(rows: list[dict[str, Any]]) -> None:
    path = OUTPUTS / "experiment_comparison.csv"
    fields = [
        "experiment", "model_path", "image_size", "filters", "optimizer", "learning_rate", "weight_decay",
        "batch_size", "dropout", "augmentation", "epochs_run", "best_epoch", "best_val_accuracy",
        "best_val_loss", "validation_threshold", "val_accuracy", "val_accuracy_at_0_5", "val_precision_macro",
        "val_recall_macro", "val_f1_macro", "val_f1_weighted", "val_dog_recall", "val_cat_recall",
        "training_seconds", "parameters", "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (dict, list, tuple)) else value for key, value in row.items()})


def _baseline_result(val_ds: tf.data.Dataset) -> dict[str, Any]:
    model_path = BASELINE_BACKUP / "models" / "dogs_cats_model.keras"
    if not model_path.is_file():
        raise SystemExit(f"Archived baseline model is missing: {model_path}")
    model = tf.keras.models.load_model(model_path)
    scores = score_model(model, val_ds)
    history_path = BASELINE_BACKUP / "outputs" / "training_history.json"
    history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.is_file() else {}
    val_loss = history.get("val_loss", [])
    val_accuracy = history.get("val_accuracy", [])
    row = {
        "experiment": "baseline_original",
        "model_path": model_path.relative_to(ROOT).as_posix(),
        "image_size": [128, 128], "filters": [32, 64, 128], "optimizer": "Adam",
        "learning_rate": 0.001, "weight_decay": 0.0, "batch_size": 32, "dropout": [0.2, 0.2, 0.3, 0.4],
        "augmentation": {"flip": True, "rotation": 0.08, "zoom": 0.10, "translation": 0.08, "brightness": 0, "contrast": 0},
        "epochs_run": len(history.get("accuracy", [])),
        "best_epoch": int(np.argmin(val_loss) + 1) if val_loss else "",
        "best_val_accuracy": float(max(val_accuracy)) if val_accuracy else "",
        "best_val_loss": float(min(val_loss)) if val_loss else "",
        "validation_threshold": scores["threshold"],
        "val_accuracy": scores["accuracy"], "val_accuracy_at_0_5": scores["accuracy_at_0_5"],
        "val_precision_macro": scores["precision_macro"], "val_recall_macro": scores["recall_macro"],
        "val_f1_macro": scores["f1_macro"], "val_f1_weighted": scores["f1_weighted"],
        "val_dog_recall": scores["dog_recall"], "val_cat_recall": scores["cat_recall"],
        "training_seconds": 0, "parameters": int(model.count_params()),
        "notes": "Existing baseline; threshold chosen using validation split only.",
    }
    del model
    tf.keras.backend.clear_session()
    return row


def _paths_for_split(split: str) -> list[Path]:
    return [path for label in CLASS_NAMES for path in sorted((DATASET / split / label).glob("*.jpg"))]


def write_validation_error_analysis(model: tf.keras.Model, dataset: tf.data.Dataset, threshold: float, out_dir: Path) -> dict[str, Any]:
    """Save validation-only predictions and image examples; never reads the test split."""
    out_dir.mkdir(parents=True, exist_ok=True)
    scored = score_model(model, dataset)
    y_true = scored["y_true"]
    p_cat = scored["p_cat"]
    y_pred = (p_cat >= threshold).astype(np.int32)
    paths = _paths_for_split("validation")
    if len(paths) != len(y_true):
        raise RuntimeError(f"Validation path/label count mismatch: {len(paths)} != {len(y_true)}")
    with (out_dir / "validation_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "actual", "predicted", "p_cat", "p_dog", "threshold", "correct"])
        for path, actual, pred, probability in zip(paths, y_true, y_pred, p_cat):
            writer.writerow([path.relative_to(ROOT).as_posix(), CLASS_NAMES[int(actual)], CLASS_NAMES[int(pred)], f"{probability:.8f}", f"{1-probability:.8f}", f"{threshold:.3f}", int(actual == pred)])

    groups = {
        "false_negative_cats": np.where((y_true == 1) & (y_pred == 0))[0],
        "false_positive_cats": np.where((y_true == 0) & (y_pred == 1))[0],
        "correct_dogs": np.where((y_true == 0) & (y_pred == 0))[0],
        "correct_cats": np.where((y_true == 1) & (y_pred == 1))[0],
    }
    for group, indices in groups.items():
        indices = sorted(indices, key=lambda i: abs(float(p_cat[i]) - threshold), reverse=True)[:12]
        columns, nrows = 4, max(1, (len(indices) + 3) // 4)
        fig, axes = plt.subplots(nrows, columns, figsize=(12, nrows * 3))
        axes = np.asarray(axes).reshape(-1)
        for ax, index in zip(axes, indices):
            with Image.open(paths[index]) as image:
                ax.imshow(image.convert("RGB"))
            ax.axis("off")
            ax.set_title(f"True {CLASS_NAMES[y_true[index]]} · Pred {CLASS_NAMES[y_pred[index]]}\nP(Cat)={p_cat[index]:.2f}", fontsize=9)
        for ax in axes[len(indices):]:
            ax.axis("off")
        fig.suptitle(f"Validation examples: {group.replace('_', ' ')} ({len(indices)} shown)")
        fig.tight_layout()
        fig.savefig(out_dir / f"{group}.png", dpi=130)
        plt.close(fig)
    validation_metrics = {key: value for key, value in scored.items() if key not in ("y_true", "p_cat")}
    validation_metrics.update({"selected_threshold": threshold, "confusion_matrix_at_selected_threshold": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()})
    (out_dir / "metrics.json").write_text(json.dumps(validation_metrics, indent=2), encoding="utf-8")
    return validation_metrics


def run_experiment(config: dict[str, Any], max_epochs: int, patience: int, seed: int) -> dict[str, Any]:
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    height, width = config["image_size"]
    train_ds = make_dataset(str(DATASET / "train"), config["batch_size"], (height, width), shuffle=True, seed=seed)
    val_ds = make_dataset(str(DATASET / "validation"), config["batch_size"], (height, width), shuffle=False, seed=seed)
    experiment_dir = OUTPUTS / "experiments" / config["name"]
    experiment_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = experiment_dir / "best_model.keras"
    model = build_model(
        image_size=(height, width), filters=config["filters"], dropout=config["dropout"],
        learning_rate=config["learning_rate"], optimizer_name=config["optimizer"],
        weight_decay=config["weight_decay"], augmentation=config["augmentation"],
    )
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(checkpoint, monitor="val_loss", mode="min", save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=patience, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(2, patience // 2), min_lr=1e-6, verbose=1),
        tf.keras.callbacks.CSVLogger(experiment_dir / "training_log.csv"),
    ]
    start = time.perf_counter()
    history = model.fit(train_ds, validation_data=val_ds, epochs=max_epochs, callbacks=callbacks, verbose=2, shuffle=False)
    elapsed = time.perf_counter() - start
    history_data = {key: [float(v) for v in values] for key, values in history.history.items()}
    (experiment_dir / "history.json").write_text(json.dumps(history_data, indent=2), encoding="utf-8")
    model = tf.keras.models.load_model(checkpoint)
    scores = score_model(model, val_ds)
    best_epoch = int(np.argmin(history_data["val_loss"]) + 1)
    best_val_acc = float(max(history_data["val_accuracy"]))
    best_val_loss = float(min(history_data["val_loss"]))
    candidate_models = EXPERIMENT_CANDIDATES
    candidate_models.mkdir(parents=True, exist_ok=True)
    candidate_path = candidate_models / f"{config['name']}.keras"
    shutil.copy2(checkpoint, candidate_path)
    write_validation_error_analysis(model, val_ds, scores["threshold"], experiment_dir / "error_analysis")
    row = {
        "experiment": config["name"], "model_path": candidate_path.relative_to(ROOT).as_posix(),
        "image_size": list(config["image_size"]), "filters": list(config["filters"]),
        "optimizer": config["optimizer"], "learning_rate": config["learning_rate"], "weight_decay": config["weight_decay"],
        "batch_size": config["batch_size"], "dropout": config["dropout"], "augmentation": config["augmentation"],
        "epochs_run": len(history_data["accuracy"]), "best_epoch": best_epoch,
        "best_val_accuracy": best_val_acc, "best_val_loss": best_val_loss,
        "validation_threshold": scores["threshold"], "val_accuracy": scores["accuracy"],
        "val_accuracy_at_0_5": scores["accuracy_at_0_5"], "val_precision_macro": scores["precision_macro"],
        "val_recall_macro": scores["recall_macro"], "val_f1_macro": scores["f1_macro"],
        "val_f1_weighted": scores["f1_weighted"], "val_dog_recall": scores["dog_recall"],
        "val_cat_recall": scores["cat_recall"], "training_seconds": round(elapsed, 2),
        "parameters": int(model.count_params()), "notes": "Selected using validation macro F1; test split not accessed.",
    }
    # Per-experiment training curves use its actual fit history.
    for key, title, filename in (("accuracy", "Accuracy", "accuracy.png"), ("loss", "Loss", "loss.png")):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(history_data[key], label="Training")
        ax.plot(history_data[f"val_{key}"], label="Validation")
        ax.set_title(f"{config['name']} · {title}"); ax.set_xlabel("Epoch"); ax.set_ylabel(title); ax.legend(); fig.tight_layout()
        fig.savefig(experiment_dir / filename, dpi=140); plt.close(fig)
    del model, train_ds, val_ds
    tf.keras.backend.clear_session()
    print(f"Completed {config['name']}: val macro F1={row['val_f1_macro']:.4f}, cat recall={row['val_cat_recall']:.4f}, threshold={row['validation_threshold']:.2f}, {row['epochs_run']} epochs, {elapsed/60:.1f} min")
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiments", nargs="*", choices=[item["name"] for item in EXPERIMENTS], default=None)
    args = parser.parse_args()
    if args.max_epochs < 1 or args.patience < 1:
        parser.error("--max-epochs and --patience must be positive")
    if not (DATASET / "train").is_dir() or not (DATASET / "validation").is_dir():
        raise SystemExit("Training/validation splits are missing. Run src/preprocessing.py first.")
    OUTPUTS.mkdir(exist_ok=True)
    selected_configs = [item for item in EXPERIMENTS if args.experiments is None or item["name"] in args.experiments]
    if not selected_configs:
        raise SystemExit("No experiments selected")

    baseline_image_size = DEFAULT_IMAGE_SIZE
    baseline_val_ds = make_dataset(str(DATASET / "validation"), batch_size=32, image_size=baseline_image_size, shuffle=False, seed=args.seed)
    rows = [_baseline_result(baseline_val_ds)]
    del baseline_val_ds
    write_comparison(rows)
    print(f"Baseline validation macro F1={rows[0]['val_f1_macro']:.4f}; cat recall={rows[0]['val_cat_recall']:.4f} (validation-only threshold={rows[0]['validation_threshold']:.2f})")
    for config in selected_configs:
        row = run_experiment(config, args.max_epochs, args.patience, args.seed)
        rows.append(row)
        write_comparison(rows)

    # Validation macro-F1 is the primary selector. Accuracy and loss break ties.
    winner = max(rows, key=lambda row: (float(row["val_f1_macro"]), float(row["val_accuracy"]), -float(row["best_val_loss"] or float("inf"))))
    candidate = CANDIDATE_MODEL
    source_model = ROOT / winner["model_path"]
    shutil.copy2(source_model, candidate)
    metadata = {
        "selected_experiment": winner["experiment"],
        "candidate_model": str(candidate.relative_to(ROOT).as_posix()),
        "production_model": "models/dogs_cats_model.keras",
        "class_mapping": {"Dog": 0, "Cat": 1},
        "output": "P(Cat)",
        "dataset": DATASET.relative_to(ROOT).as_posix(),
        "image_size": winner["image_size"],
        "filters": winner["filters"],
        "batch_size": winner["batch_size"],
        "optimizer": winner["optimizer"],
        "learning_rate": winner["learning_rate"],
        "dropout": winner["dropout"],
        "augmentation": winner["augmentation"],
        "validation_threshold": float(winner["validation_threshold"]),
        "selection_metric": "validation macro F1, then validation accuracy, then validation loss",
        "validation_metrics": {key: winner[key] for key in ("val_accuracy", "val_precision_macro", "val_recall_macro", "val_f1_macro", "val_f1_weighted", "val_dog_recall", "val_cat_recall")},
        "test_split_used_for_selection": False,
        "seed": args.seed,
    }
    CANDIDATE_METADATA.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    selected_model = tf.keras.models.load_model(candidate)
    final_val_size = tuple(winner["image_size"])
    final_val_ds = make_dataset(str(DATASET / "validation"), 32, final_val_size, shuffle=False, seed=args.seed)
    write_validation_error_analysis(selected_model, final_val_ds, float(winner["validation_threshold"]), OUTPUTS / "error_analysis" / "selected_validation")
    (OUTPUTS / "model_selection.json").write_text(json.dumps({"winner": metadata, "experiments": [{k: v for k, v in row.items() if k not in ("augmentation", "filters")} for row in rows]}, indent=2), encoding="utf-8")
    print(f"Selected {winner['experiment']} using validation only: macro F1={winner['val_f1_macro']:.4f}; cat recall={winner['val_cat_recall']:.4f}.")
    print(f"Candidate saved separately at {candidate}. Production model has not been replaced; run evaluation with --promote after selection.")


if __name__ == "__main__":
    main()
