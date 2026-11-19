# PawVision: AI Dogs & Cats Classifier

PawVision is a Streamlit app that tells you whether a photo shows a dog or a cat. It also tells you how sure it is and which parts of the photo it looked at. The production model is an ImageNet-pretrained **EfficientNetV2-B0**, fine-tuned on this project's dataset.

## Final held-out test result

The model was picked on validation macro F1. It was then scored on the 750-image test split, which was used for no decision. That split holds the same files the earlier scratch CNN was tested on.

| Metric | Original baseline | Scratch CNN | **EfficientNetV2-B0 (production)** |
|---|---:|---:|---:|
| Test accuracy | 72.27% | 74.80% | **99.60%** |
| Test loss | 0.5423 | 0.5261 | **0.0184** |
| Macro F1 | 71.74% | 74.78% | **99.60%** |
| ROC AUC | — | 0.8234 | **0.9998** |
| Cat recall | 58.67% | 77.60% | **100.00%** |
| Dog recall | 85.87% | 72.00% | **99.20%** |

The model got 747 of 750 test photos right. All 3 mistakes were dogs it called cats. See `outputs/final_comparison.md` and `outputs/classification_report.txt`.

## How the model was improved

Before, the model was a small CNN trained from scratch on 3,500 images at 128×128. It reached 74.8% on the test set. Now it uses **transfer learning**:

1. **Same split, higher resolution.** `preprocessing.py --image-size 224` rebuilds the dataset at 224×224 from the original images, using the same seed. The train, validation and test files are identical to the 128px version, so the comparison is fair.
2. **Two-stage fine-tuning** (`src/train_transfer.py`):
   - **Stage 1:** the pretrained backbone is frozen, and a new head is trained (global average pooling, then dropout, then a sigmoid). Settings: up to 5 epochs, Adam at 1e-3.
   - **Stage 2:** the top layers of the backbone are unfrozen, and the model is fine-tuned at 2e-5 for up to 8 epochs. BatchNorm stays frozen throughout. Early stopping uses validation loss.
   - Training-only augmentation: horizontal flip, small rotation and zoom, contrast.
3. **Validation-only selection.** Two backbones were compared:

| Candidate | Params | Val accuracy | Val macro F1 | Val loss | Threshold | CPU time |
|---|---:|---:|---:|---:|---:|---:|
| MobileNetV2, top 40 layers fine-tuned | 2.26 M | 99.60% | 0.9960 | 0.0156 | 0.66 | 10.7 min |
| **EfficientNetV2-B0, top 60 layers fine-tuned (selected)** | 5.92 M | **99.87%** | **0.9987** | **0.0083** | 0.50 | 11.6 min |

Every model takes RGB input in [0, 1] and does its own preprocessing inside the network, so the app and evaluator stay model-agnostic. `Dog=0`, `Cat=1`, and the output is `P(Cat)`.

**A note on the test split:** the earlier scratch CNN was evaluated on this same test split. The new model's evaluation therefore had to pass `--allow-repeat-test`. The test set was still not used for choosing the backbone, the hyperparameters or the threshold.

The scratch CNN, its code and its outputs are kept in `backups/scratch_cnn_20261119/`. The original baseline is in `backups/baseline_20261117/`.

## Dataset

The source is Hugging Face [`shravya11/dog-cat-dataset`](https://huggingface.co/datasets/shravya11/dog-cat-dataset). From its train split, we take 2,500 cats and 2,500 dogs. These go into fixed-seed, disjoint splits of 70/15/15: 3,500 train, 750 validation and 750 test. The source test split is not used.

## Project structure

```text
dogs-vs-cats/
├── app.py                      # Streamlit app (entry point)
├── requirements.txt
├── .streamlit/config.toml      # Light theme, upload limit
├── src/
│   ├── paths.py                # All project paths, resolved from the repo root
│   ├── preprocessing.py        # Builds the balanced 70/15/15 split from the Hugging Face cache
│   ├── model.py                # tf.data input pipeline + scratch CNN architecture
│   ├── train_scratch.py        # Scratch-CNN experiments (validation-only selection)
│   ├── train_transfer.py       # MobileNetV2 / EfficientNetV2-B0 fine-tuning (production)
│   └── evaluate.py             # One-time test evaluation, charts, promotion to production
├── notebooks/
│   └── model_training.ipynb    # Interactive version of the scratch pipeline
├── data/                       # Generated, git-ignored
│   ├── processed_128/{train,validation,test}/{Cat,Dog}/
│   └── processed_224/{train,validation,test}/{Cat,Dog}/
├── models/
│   ├── dogs_cats_model.keras   # Production model (git-ignored)
│   ├── model_metadata.json     # Threshold, input size, selection record
│   ├── pet_gate_imagenet.keras # "Is this a pet?" check used by the app (git-ignored)
│   ├── imagenet_class_index.json
│   └── experiment_candidates/  # Every trained candidate (git-ignored)
├── outputs/                    # Metrics, reports, charts, per-experiment histories
└── backups/                    # Archived earlier versions; evaluate.py and train_scratch.py read their metrics
```

Datasets and model weights are git-ignored because of their size, so a fresh clone has to rebuild them with the steps below. Every script resolves paths from the repository root, so it can be run from any directory.

## Run on Windows

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Reproduce the production model. Training takes about 25 minutes on CPU, and the first run downloads ImageNet weights:

```powershell
python src\preprocessing.py --per-class 2500 --seed 42 --image-size 224 --destination data\processed_224
python src\train_transfer.py
python src\evaluate.py --promote --allow-repeat-test
```

`train_transfer.py` also saves `models/pet_gate_imagenet.keras`. This is the full ImageNet MobileNetV2, which the app uses to warn when a photo doesn't look like a dog or cat.

The scratch-CNN pipeline still works. It writes `data/processed_128` by default:

```powershell
python src\preprocessing.py --per-class 2500 --seed 42
python src\train_scratch.py --max-epochs 20 --patience 4 --seed 42
python src\evaluate.py --allow-repeat-test
```

Leave out `--promote` here unless you want the scratch CNN to replace the production model.

## Run PawVision

```powershell
streamlit run app.py
```

The app has three pages, reached from the top navigation bar:

- **Classify**
  - Add a photo in one of three ways: upload one or more photos, pick a built-in example from the test set, or use your camera.
  - The result appears straight away. You don't need to press a classify button.
  - Each result shows a plain-language confidence label (very confident, fairly confident, or not sure) and a single dog-vs-cat bar.
  - A toggle shows **what the model focused on**, as a heatmap over the photo.
  - It warns you if a photo might not show a dog or cat (it also says what the photo looks like instead), or if a photo might contain both animals.
  - If you add several photos, you get a results grid and can download the results as a CSV file.
- **Model results:** results explained in plain words (for example, "it got 747 of 750 right"). It also has a before-and-after comparison with the scratch CNN, a colour-coded confusion matrix, the mistakes it made, and an expander with the technical charts.
- **How it works:** the four-step pipeline, how transfer learning and the fair testing were done, the dataset split, its limits, and the privacy note.

Photos are processed in memory and never saved. The app forces Streamlit's light theme (`.streamlit/config.toml`) so the custom styling stays readable.
