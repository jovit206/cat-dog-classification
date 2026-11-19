"""Single source of truth for PawVision project paths.

Every path is absolute and derived from this file's location, so scripts work no matter
which directory they are launched from.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data"
DATASET_128 = DATA / "processed_128"  # Scratch CNN input (128 x 128).
DATASET_224 = DATA / "processed_224"  # Transfer-learning input (224 x 224).

MODELS = ROOT / "models"
PRODUCTION_MODEL = MODELS / "dogs_cats_model.keras"
PRODUCTION_METADATA = MODELS / "model_metadata.json"
CANDIDATE_MODEL = MODELS / "candidate_dogs_cats_model.keras"
CANDIDATE_METADATA = MODELS / "candidate_metadata.json"
EXPERIMENT_CANDIDATES = MODELS / "experiment_candidates"

OUTPUTS = ROOT / "outputs"

BACKUPS = ROOT / "backups"
BASELINE_BACKUP = BACKUPS / "baseline_20261117"
SCRATCH_BACKUP = BACKUPS / "scratch_cnn_20261119"


def relative(path: Path) -> str:
    """Project-relative POSIX path for metadata files (falls back to the absolute path)."""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)
