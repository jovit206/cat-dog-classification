# PawVision – Dogs vs Cats Image Classification

PawVision is a Streamlit application backed by an original TensorFlow/Keras CNN that predicts whether an uploaded image contains a dog or a cat. Its preprocessing pipeline creates a manageable, balanced sample from the locally cached Hugging Face Dogs vs Cats dataset.

## Problem and objectives

Distinguish dogs from cats in RGB photographs, show the predicted class and confidence, and provide a reproducible training and evaluation workflow. The model is built and trained from scratch; it uses no pretrained model, external prediction API, or simulated metrics.

## Technology

Python, TensorFlow/Keras, CNN, NumPy, Pandas, scikit-learn, Matplotlib, Seaborn, Pillow, Streamlit, and custom CSS.

## Dataset

The preprocessing pipeline uses the Hugging Face `shravya11/dog-cat-dataset` imagefolder. Its source layout has `train/cats/part_*`, `train/dogs/part_*`, `test/cats`, and `test/dogs`. The cached dataset metadata reports 19,989 training images (10,000 cats and 9,989 dogs) and 5,000 test images (2,500 of each). PawVision reads only the Hugging Face training split, selecting 2,500 images per class; it leaves the source dataset, including its test split, unchanged.

The preparation script samples exactly 2,500 images per class by default, then creates disjoint splits with 1,750 images per class for training and 375 per class each for validation and testing. It writes separate 128 × 128 RGB JPEGs under `dataset/` and records counts and a source manifest under `outputs/`. The fixed default seed is 42. `Dog` maps to 0 and `Cat` maps to 1 throughout the loader and sigmoid output.

## Windows PowerShell setup

Use Python 3.10–3.12 (the current host has Python 3.12). In PowerShell, from this project directory:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process Bypass` in that terminal and activate again. TensorFlow's Windows package is CPU-only; GPU training on Windows requires WSL2 with a supported NVIDIA/CUDA setup.

## Dataset preparation

The script auto-detects the Hugging Face snapshot under `%USERPROFILE%\.cache\huggingface\hub\datasets--shravya11--dog-cat-dataset\snapshots`. Run:

```powershell
python src\preprocessing.py --per-class 2500 --seed 42
```

If the dataset is extracted elsewhere, pass the root folder containing `train\cats` and `train\dogs` with `--source C:\path\to\dog-cat-dataset`. The processed image folders are `dataset\train`, `dataset\validation`, and `dataset\test`, each with `Dog` and `Cat` subfolders. The original Hugging Face images remain untouched.

## Model and training

The model takes 128 × 128 × 3 RGB tensors normalized to [0, 1]. It applies on-the-fly rotation, horizontal flipping, zoom, and translation; three Conv2D blocks (32, 64, 128 filters) with batch normalization, ReLU, max pooling and dropout; global average pooling; a dense layer; and a sigmoid output. Adam, binary cross-entropy, and accuracy are used. Training uses batch size 32, early stopping, best-validation-accuracy checkpointing, and learning-rate reduction on plateau.

```powershell
python src\train.py --epochs 15 --batch-size 32
```

The best checkpoint is saved to `models\dogs_cats_model.keras`, and genuine epoch history to `outputs\training_history.json`.

## Evaluation

```powershell
python src\evaluate.py
```

Evaluation reloads the saved model and runs it on the held-out test split. It saves test loss, accuracy, macro precision, recall, and F1 to `outputs\metrics.json`; the classification report, confusion matrix, and accuracy/loss charts are saved in `outputs`.

**Evaluation results:** Test accuracy 72.27%, test loss 0.5423, macro precision 74.05%, macro recall 72.27%, and macro F1 score 71.74% on 750 held-out images. The app reads the generated `outputs\metrics.json` and supporting report and charts.

## Launch the application

```powershell
streamlit run app.py
```

Use the sidebar to open Home, Classify an Image, Model Performance, How It Works, or About PawVision. On the classifier page, upload JPG/JPEG/PNG and click **Classify image**. Predictions use the saved CNN directly. A missing model or missing evaluation files is reported clearly.

## Project structure

```text
dogs-vs-cats/
├── dataset/{train,validation,test}/
├── models/
├── notebooks/model_training.ipynb
├── outputs/
├── src/{preprocessing,model,train,evaluate}.py
├── app.py
├── requirements.txt
└── README.md
```
