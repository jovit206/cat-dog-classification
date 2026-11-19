"""Build a balanced, resized PawVision dataset from a local Hugging Face imagefolder."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import uuid
from pathlib import Path

from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "dataset"
SUMMARY_DIR = ROOT / "outputs"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
LABELS = {"cats": "Cat", "dogs": "Dog"}
SPLIT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}
IMAGE_SIZE = (128, 128)


def _is_source_root(path: Path) -> bool:
    return all((path / "train" / folder).is_dir() for folder in LABELS)


def resolve_source(source: Path | None = None) -> Path:
    """Resolve either the dataset root or this user's cached HF repository snapshot."""
    if source is not None:
        source = source.expanduser().resolve()
        if _is_source_root(source):
            return source
        raise FileNotFoundError(
            f"Expected train/cats and train/dogs under {source}. "
            "Pass the extracted Hugging Face dataset root with --source."
        )

    snapshots = Path.home() / ".cache" / "huggingface" / "hub" / "datasets--shravya11--dog-cat-dataset" / "snapshots"
    matches = [p for p in snapshots.iterdir() if p.is_dir() and _is_source_root(p)] if snapshots.is_dir() else []
    if len(matches) == 1:
        return matches[0]
    if matches:
        return max(matches, key=lambda p: p.stat().st_mtime)
    raise FileNotFoundError(
        "Could not find the cached Hugging Face dataset. Expected it under "
        f"{snapshots}, or pass its extracted root with --source."
    )


def _image_files(folder: Path) -> list[Path]:
    return sorted(
        (path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda path: path.as_posix().lower(),
    )


def _count_source(source: Path) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for split in ("train", "test"):
        split_dir = source / split
        if not split_dir.is_dir():
            continue
        counts[split] = {
            label: len(_image_files(split_dir / folder)) if (split_dir / folder).is_dir() else 0
            for folder, label in LABELS.items()
        }
    return counts


def _make_assignments(source: Path, per_class: int, seed: int) -> dict[str, list[tuple[Path, str]]]:
    if per_class < 1:
        raise ValueError("per_class must be a positive integer")
    n_train = int(per_class * SPLIT_RATIOS["train"])
    n_validation = int(per_class * SPLIT_RATIOS["validation"])
    n_test = per_class - n_train - n_validation
    split_sizes = {"train": n_train, "validation": n_validation, "test": n_test}
    assignments: dict[str, list[tuple[Path, str]]] = {name: [] for name in SPLIT_RATIOS}

    for class_index, (folder, label) in enumerate(LABELS.items()):
        candidates = _image_files(source / "train" / folder)
        if len(candidates) < per_class:
            raise ValueError(f"Need {per_class} {label} images in source train split; found {len(candidates)}")
        rng = random.Random(seed + class_index)
        selected = rng.sample(candidates, per_class)
        rng.shuffle(selected)
        offset = 0
        for split, size in split_sizes.items():
            assignments[split].extend((path, label) for path in selected[offset : offset + size])
            offset += size
    return assignments


def _relative_name(path: Path, source: Path) -> str:
    # Include Hugging Face part_* folders to avoid filename collisions after flattening.
    relative = path.relative_to(source / "train").with_suffix("")
    return "__".join(relative.parts) + ".jpg"


def _summary_rows(counts: dict[str, dict[str, int]]) -> list[dict[str, int | str]]:
    rows: list[dict[str, int | str]] = []
    for split in (*SPLIT_RATIOS, "total"):
        dog_count = sum(int(counts[name]["Dog"]) for name in SPLIT_RATIOS) if split == "total" else int(counts[split]["Dog"])
        cat_count = sum(int(counts[name]["Cat"]) for name in SPLIT_RATIOS) if split == "total" else int(counts[split]["Cat"])
        rows.append({"split": split, "Dog": dog_count, "Cat": cat_count, "total": dog_count + cat_count})
    return rows


def _publish_splits(staging: Path, destination: Path) -> None:
    """Replace only the generated split folders, rolling back if publishing fails."""
    destination.mkdir(parents=True, exist_ok=True)
    backup = destination.parent / f".pawvision_backup_{uuid.uuid4().hex}"
    backup.mkdir()
    moved_old: list[str] = []
    moved_new: list[str] = []
    try:
        for split in SPLIT_RATIOS:
            target = destination / split
            if target.exists():
                target.rename(backup / split)
                moved_old.append(split)
            (staging / split).rename(target)
            moved_new.append(split)
    except Exception:
        for split in reversed(moved_new):
            shutil.rmtree(destination / split, ignore_errors=True)
        for split in reversed(moved_old):
            old = backup / split
            if old.exists():
                old.rename(destination / split)
        raise
    finally:
        shutil.rmtree(backup, ignore_errors=True)


def prepare_dataset(
    source: Path | None = None,
    destination: Path = DESTINATION,
    per_class: int = 2500,
    seed: int = 42,
    summary_dir: Path = SUMMARY_DIR,
    image_size: tuple[int, int] = IMAGE_SIZE,
) -> dict[str, object]:
    """Select a balanced sample, make disjoint splits, resize, and write a summary.

    Original Hugging Face images are read-only inputs. Processed RGB JPEGs are
    written under ``destination/{train,validation,test}/{Dog,Cat}``.
    """
    source_root = resolve_source(source)
    source_counts = _count_source(source_root)
    assignments = _make_assignments(source_root, per_class, seed)
    staging = destination.parent / f".pawvision_staging_{uuid.uuid4().hex}"
    staging.mkdir(parents=True)
    output_counts: dict[str, dict[str, int]] = {split: {label: 0 for label in LABELS.values()} for split in SPLIT_RATIOS}
    seen_paths: set[Path] = set()
    seen_digests: set[str] = set()
    manifest: list[dict[str, str | int]] = []

    try:
        for split, rows in assignments.items():
            for label in LABELS.values():
                (staging / split / label).mkdir(parents=True, exist_ok=True)
            for source_image, label in rows:
                resolved = source_image.resolve()
                if resolved in seen_paths:
                    raise RuntimeError(f"Source image assigned more than once: {source_image}")
                seen_paths.add(resolved)

                try:
                    with Image.open(source_image) as image:
                        image.load()
                        rgb = image.convert("RGB").resize(image_size, Image.Resampling.LANCZOS)
                        digest = hashlib.sha256(source_image.read_bytes()).hexdigest()
                        if digest in seen_digests:
                            raise RuntimeError(f"Duplicate image content selected from more than one source: {source_image}")
                        seen_digests.add(digest)
                        output_name = _relative_name(source_image, source_root)
                        target = staging / split / label / output_name
                        rgb.save(target, format="JPEG", quality=95, optimize=True)
                except (OSError, UnidentifiedImageError) as exc:
                    raise ValueError(f"Could not decode selected image {source_image}: {exc}") from exc

                output_counts[split][label] += 1
                manifest.append({
                    "split": split,
                    "class": label,
                    "source_split": "train",
                    "source_path": source_image.relative_to(source_root).as_posix(),
                    "output_path": target.relative_to(staging).as_posix(),
                    "sha256": digest,
                })

        expected_sizes = {
            "train": int(per_class * 0.70),
            "validation": int(per_class * 0.15),
            "test": per_class - int(per_class * 0.70) - int(per_class * 0.15),
        }
        for split, size in expected_sizes.items():
            if output_counts[split] != {label: size for label in LABELS.values()}:
                raise RuntimeError(f"Unexpected class balance in {split}: {output_counts[split]}")
        if len(seen_paths) != 2 * per_class or len(seen_digests) != 2 * per_class:
            raise RuntimeError("Selected image paths or image contents are not unique across splits")

        # Verify every processed file before replacing any existing output split.
        for split, rows in assignments.items():
            for source_image, label in rows:
                target = staging / split / label / _relative_name(source_image, source_root)
                with Image.open(target) as image:
                    if image.mode != "RGB" or image.size != tuple(image_size):
                        raise RuntimeError(f"Processed image failed RGB/size verification: {target}")

        _publish_splits(staging, destination)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    rows = _summary_rows(output_counts)
    summary_dir.mkdir(parents=True, exist_ok=True)
    with (summary_dir / "dataset_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "Dog", "Cat", "total"])
        writer.writeheader()
        writer.writerows(rows)
    summary: dict[str, object] = {
        "source": str(source_root),
        "source_split_counts": source_counts,
        "selected_from": "train",
        "seed": seed,
        "image_size": list(image_size),
        "color_mode": "RGB",
        "normalization_for_training": "divide pixel values by 255 to map to [0, 1]",
        "requested_per_class": per_class,
        "split_ratios": SPLIT_RATIOS,
        "counts": output_counts,
        "summary_table": rows,
        "unique_source_paths": len(seen_paths),
        "unique_source_image_sha256": len(seen_digests),
    }
    (summary_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (summary_dir / "dataset_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "class", "source_split", "source_path", "output_path", "sha256"])
        writer.writeheader()
        writer.writerows(manifest)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Hugging Face dataset root containing train/cats and train/dogs (auto-detected from local cache by default)")
    parser.add_argument("--destination", type=Path, default=DESTINATION)
    parser.add_argument("--per-class", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE[0], help="Square output size in pixels (128 for the scratch CNN, 224 for the pretrained model)")
    args = parser.parse_args()
    summary = prepare_dataset(args.source, args.destination, args.per_class, args.seed, image_size=(args.image_size, args.image_size))
    print(json.dumps(summary["summary_table"], indent=2))
    print(f"Seed: {args.seed}; processed images: {2 * args.per_class}; output: {args.destination}")
    print(f"Summary: {SUMMARY_DIR / 'dataset_summary.csv'}")


if __name__ == "__main__":
    main()
