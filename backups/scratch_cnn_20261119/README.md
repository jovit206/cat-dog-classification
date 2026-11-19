# PawVision — AI Dogs & Cats Classifier

PawVision is a Streamlit application backed by a custom TensorFlow/Keras CNN trained from scratch. It accepts a dog or cat photo, preprocesses it, and displays the model's class scores and prediction.

## Final held-out test result

The model was selected by validation macro F1, then evaluated once on the untouched 750-image test split. The selected model used a Cat decision threshold of 0.53 chosen from validation predictions.

| Metric | Original baseline | Selected model | Change |
|---|---:|---:|---:|
| Test accuracy | 72.27% | **74.80%** | +2.53 percentage points |
| Test loss | 0.5423 | **0.5261** | −0.0162 |
| Macro precision | 74.05% | **74.88%** | +0.83 percentage points |
| Macro recall | 72.27% | **74.80%** | +2.53 percentage points |
| Macro F1 | 71.74% | **74.78%** | +3.04 percentage points |
| Weighted F1 | — | **74.78%** | — |
| ROC AUC | — | **0.8234** | — |
| Average precision (Cat positive) | — | **0.8044** | — |

Cat recall increased from **58.67% to 77.60%**. Dog recall decreased from 85.87% to 72.00%, so the result is a more balanced classifier rather than an improvement for every individual class metric. The final report and comparison are in `outputs/classification_report.txt` and `outputs/final_comparison.md`.

| Class | Precision, baseline → selected | Recall, baseline → selected | F1, baseline → selected |
|---|---:|---:|---:|
| Dog | 67.51% → 76.27% | 85.87% → 72.00% | 75.59% → 74.07% |
| Cat | 80.59% → 73.48% | 58.67% → 77.60% | 67.90% → 75.49% |

## Dataset and split

The source is Hugging Face [`shravya11/dog-cat-dataset`](https://huggingface.co/datasets/shravya11/dog-cat-dataset). Preprocessing samples 2,500 cats and 2,500 dogs from its training split, then makes fixed-seed, disjoint 70/15/15 train, validation, and test splits. The source Hugging Face test split is not used. The selected images are saved as separate 128×128 RGB files under `dataset/`; original cached images remain untouched. `Dog=0`, `Cat=1` and the sigmoid output is `P(Cat)`.

The local summary and per-image source manifest are `outputs/dataset_summary.csv`, `outputs/dataset_summary.json`, and `outputs/dataset_manifest.csv`.

## CNN and training

The winning model is `cnn128_adamw_mild_aug`: a stride-2 convolution stem and four progressively wider convolution blocks (32, 64, 128, 192 filters), with batch normalization, ReLU, max pooling, dropout, global average pooling, a dense layer, and sigmoid output. It was trained from scratch with AdamW (learning rate 0.0003, weight decay 0.0001), batch size 32, and early stopping. Training-only augmentation uses horizontal flip, small rotation and zoom, translation, brightness, and contrast changes. Validation and test images receive no augmentation.

Four candidate configurations were compared using validation macro F1 and a decision threshold tuned on validation only. No test predictions were used to choose architecture, hyperparameters, or threshold.

| Experiment | Input | Optimizer | Batch | Best validation accuracy | Best validation loss | Val macro F1 | Val Cat recall |
|---|---:|---|---:|---:|---:|---:|---:|
| Adam, mild augmentation | 128 | Adam | 32 | 73.33% | 0.5455 | 0.7375 | 83.73% |
| **AdamW, mild augmentation (selected)** | 128 | AdamW | 32 | **74.40%** | **0.5367** | **0.7410** | 77.60% |
| Higher resolution | 160 | AdamW | 32 | 63.20% | 0.6681 | 0.7055 | 77.07% |
| Wider filters and more dropout | 128 | AdamW | 24 | 71.07% | 0.5592 | 0.7173 | 78.93% |

The 160×160 experiment took about six minutes and performed worse on validation, so the production model remains at 128×128. Detailed settings and timing are in `outputs/experiment_comparison.csv`; per-run histories and validation error examples are in `outputs/experiments/` and `outputs/error_analysis/`.

## Run on Windows

From PowerShell in this project directory:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The dataset is auto-detected from `%USERPROFILE%\.cache\huggingface\hub\datasets--shravya11--dog-cat-dataset\snapshots`. Prepare it if needed:

```powershell
python src\preprocessing.py --per-class 2500 --seed 42
```

Run the validation-selected experiments:

```powershell
python src\train.py --max-epochs 20 --patience 4 --seed 42
```

This records `outputs/experiment_comparison.csv` and stages the validation winner at `models/candidate_dogs_cats_model.keras`. It does not replace the production model.

Run the one-time final test evaluation and promote the selected model:

```powershell
python src\evaluate.py --model models\candidate_dogs_cats_model.keras --metadata models\candidate_metadata.json --promote
```

The evaluator scores the test split in one pass, writes the classification report, confusion matrix, ROC curve, precision-recall curve, test metrics, final comparison, and final test error examples under `outputs/`, then promotes the already validation-selected model. It refuses a repeat final test run unless `--allow-repeat-test` is explicitly passed. The original model, scripts, and evaluation files are preserved under `backups/baseline_20261117/`.

## Run PawVision

```powershell
streamlit run app.py
```

The app provides Home, Classify an Image, Model Performance, How It Works, and About PawVision pages. It loads `models/dogs_cats_model.keras`, picks up a newly promoted model through its file-version cache key, and uses the validation-selected threshold in `models/model_metadata.json`. Confidence is the selected class score; the Home page's test accuracy is a separate overall metric. Low-confidence predictions prompt the user to try a clearer image.
