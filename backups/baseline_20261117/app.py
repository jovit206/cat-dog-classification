"""PawVision: a friendly Streamlit interface for the saved Dogs vs Cats CNN."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "dogs_cats_model.keras"
DATASET_PATH = ROOT / "dataset"
OUTPUTS_PATH = ROOT / "outputs"
IMAGE_SIZE = (128, 128)
PAGES = ["Home", "Classify an Image", "Model Performance", "How It Works", "About PawVision"]

st.set_page_config(page_title="PawVision | AI Pet Classifier", page_icon="🐾", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
    :root { --ink:#17243b; --muted:#6f7d93; --blue:#476ee8; --blue-dark:#2f53c5; --teal:#3da99a; --line:#e7ecf4; }
    html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
    .stApp { background:radial-gradient(ellipse at 88% 0%,#e9f4ff 0,transparent 38%),linear-gradient(135deg,#f7f9fd 0%,#f8f9fc 60%,#fff8f2 100%); color:var(--ink); }
    h1,h2,h3 { font-family:'Manrope','DM Sans',sans-serif; color:var(--ink); letter-spacing:-.025em; }
    h1 { font-weight:800; } h2 { font-weight:750; }
    [data-testid="stSidebar"] { background:#fff; border-right:1px solid var(--line); }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color:var(--muted); }
    .brand { display:flex; align-items:center; gap:.7rem; padding:.35rem .2rem 1.2rem; }
    .brand-mark { width:44px;height:44px;display:grid;place-items:center;border-radius:14px;background:linear-gradient(135deg,#4166df,#72b9d2);font-size:1.45rem;box-shadow:0 7px 16px #476ee82e; }
    .brand-name { font:800 1.22rem 'Manrope',sans-serif;color:#1d2a42;line-height:1.1; }
    .brand-sub { font-size:.72rem;color:#8190a6;margin-top:.22rem; }
    .eyebrow { color:#4e6fd2;text-transform:uppercase;letter-spacing:.12em;font-weight:700;font-size:.74rem;margin-bottom:.38rem; }
    .hero { position:relative; overflow:hidden; padding:2rem 2.25rem; border-radius:24px; color:#fff; background:linear-gradient(112deg,#1d315a 0%,#3657a7 50%,#5797b7 100%); box-shadow:0 18px 45px #263f7026; margin:0 0 1.35rem; }
    .hero:after { content:'🐾'; position:absolute;right:4%;top:8%;font-size:7rem;opacity:.10;transform:rotate(-20deg); }
    .hero h1 { color:#fff; font-size:clamp(2rem,4vw,3rem); margin:0; }
    .hero p { color:#e3ecff; font-size:1.02rem;max-width:660px;margin:.6rem 0 0;line-height:1.65; }
    .hero-chip { display:inline-block;background:#ffffff20;border:1px solid #ffffff35;border-radius:99px;padding:.33rem .7rem;font-size:.76rem;margin-bottom:.85rem; }
    .surface { background:#fff;border:1px solid var(--line);border-radius:18px;padding:1.15rem 1.25rem;box-shadow:0 8px 26px #273d6810;height:100%; }
    .metric-label { color:var(--muted);font-size:.83rem;font-weight:600; }
    .metric-value { color:#263d70;font:800 1.8rem 'Manrope',sans-serif;margin:.25rem 0 .05rem; }
    .metric-hint { color:#8b97a9;font-size:.75rem; }
    .section-kicker { color:#4e6fd2;font-size:.75rem;font-weight:700;text-transform:uppercase;letter-spacing:.1em;margin:1.25rem 0 .3rem; }
    .soft-note { color:var(--muted);font-size:.94rem;line-height:1.65; }
    .step-card { background:#fff;border:1px solid var(--line);border-radius:16px;padding:1rem;height:100%; }
    .step-num { display:grid;place-items:center;background:#edf2ff;color:#365dcf;border-radius:11px;width:35px;height:35px;font-weight:800;margin-bottom:.65rem; }
    .step-title { font-weight:700;color:#26334b;margin-bottom:.25rem; }
    .step-copy { color:var(--muted);font-size:.83rem;line-height:1.5; }
    .result-card { text-align:center;border:1px solid #dce7fb;border-radius:20px;padding:1.5rem;background:linear-gradient(145deg,#f5f8ff,#f0fbf9); }
    .result-animal { font-size:3.2rem;line-height:1.15; }
    .result-label { color:#61728d;font-size:.8rem;text-transform:uppercase;letter-spacing:.1em;font-weight:700;margin-top:.5rem; }
    .result-class { font:800 2.15rem 'Manrope',sans-serif;color:#25458c;margin:.1rem 0; }
    .probability-row { display:flex;justify-content:space-between;color:#59677d;font-size:.84rem;margin:.65rem 0 .25rem; }
    .upload-box { border:1px dashed #b7c6e5;border-radius:18px;padding:1rem;background:#fbfcff; }
    .footer { color:#9aa5b5;font-size:.78rem;border-top:1px solid var(--line);padding-top:1rem;margin-top:2.5rem; }
    div.stButton > button { border-radius:12px;border:1px solid #dce3ef;background:#fff;color:#30405c;font-weight:700;min-height:2.75rem;transition:all .18s ease; }
    div.stButton > button:hover { border-color:#7b96e8;color:#294eb9;box-shadow:0 6px 16px #3859aa1c;transform:translateY(-1px); }
    div.stButton > button[kind="primary"] { color:#fff;background:linear-gradient(110deg,var(--blue),var(--blue-dark));border:none;box-shadow:0 8px 19px #4367d532; }
    div.stButton > button[kind="primary"]:hover { color:#fff;box-shadow:0 10px 22px #4367d54b; }
    [data-testid="stFileUploader"] { border-radius:16px; }
    @media (max-width:700px) { .hero {padding:1.45rem;border-radius:18px}.hero:after {font-size:4rem;right:3%;top:5%}.metric-value {font-size:1.45rem}.surface {padding:1rem} }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_model():
    """Load and retain the user's saved CNN for the lifetime of the Streamlit process."""
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data(show_spinner=False)
def read_json(path_string: str) -> dict:
    path = Path(path_string)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def count_images(split: str) -> int:
    folder = DATASET_PATH / split
    return sum(1 for path in folder.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png"}) if folder.exists() else 0


def dataset_totals() -> dict[str, int]:
    summary = read_json(str(OUTPUTS_PATH / "dataset_summary.json"))
    counts = summary.get("counts", {})
    if counts:
        return {split: sum(counts.get(split, {}).values()) for split in ("train", "validation", "test")}
    return {split: count_images(split) for split in ("train", "validation", "test")}


def metric_value(metrics: dict, key: str, digits: int = 2, percent: bool = False) -> str:
    value = metrics.get(key)
    if not isinstance(value, (float, int)):
        return "Not available"
    return f"{value * 100:.{digits}f}%" if percent else f"{value:.{digits}f}"


def metric_card(label: str, value: str, hint: str = "") -> None:
    st.markdown(
        f'<div class="surface"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-hint">{hint}</div></div>',
        unsafe_allow_html=True,
    )


def step_card(number: int, title: str, copy: str) -> None:
    st.markdown(
        f'<div class="step-card"><div class="step-num">{number:02d}</div><div class="step-title">{title}</div><div class="step-copy">{copy}</div></div>',
        unsafe_allow_html=True,
    )


def navigate(page: str) -> None:
    st.session_state["page_nav"] = page


if "page_nav" not in st.session_state:
    st.session_state["page_nav"] = "Home"
st.session_state.setdefault("uploader_version", 0)

with st.sidebar:
    st.markdown('<div class="brand"><div class="brand-mark">🐾</div><div><div class="brand-name">PawVision</div><div class="brand-sub">AI pet image classifier</div></div></div>', unsafe_allow_html=True)
    st.markdown("<div class='section-kicker'>Workspace</div>", unsafe_allow_html=True)
    page = st.radio("Navigate", PAGES, key="page_nav", label_visibility="collapsed")
    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("A simple, friendly demo powered by your trained CNN.")
    if MODEL_PATH.is_file():
        st.success("CNN model ready", icon="✅")
    else:
        st.warning("Model file not found", icon="⚠️")

metrics = read_json(str(OUTPUTS_PATH / "metrics.json"))
dataset_counts = dataset_totals()
train_count = dataset_counts["train"]
total_images = sum(dataset_counts.values())
model_ready = MODEL_PATH.is_file()


if page == "Home":
    st.markdown(
        '<div class="hero"><span class="hero-chip">🐶 &nbsp;🐱 &nbsp; A friendly AI demo</span><h1>See what the model sees.</h1><p>Upload a photo and PawVision will use its trained image model to decide whether it sees a dog or a cat, then show how confident it is.</p></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="eyebrow">Welcome to PawVision</div>', unsafe_allow_html=True)
    st.write("A clear, hands-on way to explore image classification. No machine learning background needed: add a picture, click classify, and see the result.")

    cols = st.columns(4, gap="medium")
    metric_values = [
        ("Training images", f"{train_count:,}", "Balanced images used to learn"),
        ("Classes", "2", "Dog and cat"),
        ("Test accuracy", metric_value(metrics, "test_accuracy", percent=True), "Measured on 750 unseen images"),
        ("Model type", "Custom CNN", "Trained from scratch"),
    ]
    for col, (label, value, hint) in zip(cols, metric_values):
        with col:
            metric_card(label, value, hint)

    st.markdown('<div class="section-kicker">Ready when you are</div>', unsafe_allow_html=True)
    cta, tip = st.columns([1, 2], vertical_alignment="center")
    with cta:
        st.button("Start classification  →", key="start_classification", type="primary", on_click=navigate, args=("Classify an Image",), width="stretch")
    with tip:
        st.markdown('<div class="soft-note">Choose a clear photo of one animal for the best chance of a useful prediction. PawVision is a classroom demonstration, so results can be wrong.</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-kicker">A simple three-step process</div>', unsafe_allow_html=True)
    a, b, c = st.columns(3, gap="medium")
    with a: step_card(1, "Add a photo", "Choose a JPG, JPEG, or PNG image from your device.")
    with b: step_card(2, "Let PawVision look", "The image is resized and passed to the saved CNN model.")
    with c: step_card(3, "See the result", "Get a dog or cat prediction and the model's confidence score.")

elif page == "Classify an Image":
    st.markdown('<div class="eyebrow">Image classification</div>', unsafe_allow_html=True)
    st.title("What’s in your photo?")
    st.markdown('<div class="soft-note">Upload a photo of a dog or cat, preview it, then click <b>Classify image</b>. Your photo is processed by the saved PawVision CNN.</div>', unsafe_allow_html=True)
    if not model_ready:
        st.error(f"The trained model file is missing: `{MODEL_PATH}`. Restore the model to enable predictions.")

    uploader_key = f"pawvision_upload_{st.session_state['uploader_version']}"
    st.markdown('<div class="section-kicker">Your image</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Choose a JPG, JPEG, or PNG image", type=["jpg", "jpeg", "png"], key=uploader_key, help="Use a clear photo that contains one dog or one cat.")

    if uploaded is None:
        st.markdown('<div class="soft-note">No image selected yet. Your image preview and classification controls will appear here after upload.</div>', unsafe_allow_html=True)
    else:
        file_bytes = uploaded.getvalue()
        image_digest = hashlib.sha256(file_bytes).hexdigest()
        try:
            with Image.open(uploaded) as opened:
                opened.load()
                preview = ImageOps.exif_transpose(opened).convert("RGB")
            left, right = st.columns([1.05, 1], gap="large", vertical_alignment="top")
            with left:
                st.image(preview, caption=uploaded.name, width="stretch", alt="Uploaded photo preview")
            with right:
                st.markdown('<div class="surface"><div class="eyebrow">Ready to analyze</div><h3 style="margin-top:.2rem">One click to a prediction</h3><div class="soft-note">We’ll convert your photo to RGB, resize it to 128 × 128 pixels, normalize the values, then send it to the trained CNN.</div></div>', unsafe_allow_html=True)
                classify = st.button("Classify image", key="classify_image", type="primary", disabled=not model_ready, width="stretch")
                current_result = st.session_state.get("classification_result")
                if classify:
                    try:
                        with st.spinner("PawVision is looking at your photo…"):
                            rgb = ImageOps.exif_transpose(preview).resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
                            batch = np.asarray(rgb, dtype=np.float32)[None, ...] / 255.0
                            # The final sigmoid output is P(Cat), matching the training label map.
                            probability_cat = float(np.asarray(load_model().predict(batch, verbose=0)).reshape(-1)[0])
                            if not np.isfinite(probability_cat):
                                raise ValueError("The model returned an invalid probability.")
                            probability_cat = float(np.clip(probability_cat, 0.0, 1.0))
                            label = "Cat" if probability_cat >= 0.5 else "Dog"
                            confidence = probability_cat if label == "Cat" else 1.0 - probability_cat
                            current_result = {
                                "digest": image_digest,
                                "label": label,
                                "confidence": confidence,
                                "cat_probability": probability_cat,
                                "dog_probability": 1.0 - probability_cat,
                            }
                            st.session_state["classification_result"] = current_result
                    except FileNotFoundError:
                        st.error("PawVision could not find the trained CNN model. Restore the model file and try again.")
                    except (UnidentifiedImageError, OSError, ValueError) as exc:
                        st.error(f"PawVision couldn’t classify this image. Please try another JPG, JPEG, or PNG image. Details: {exc}")
                    except Exception as exc:
                        st.error(f"The saved CNN could not complete this prediction. Check the model file and try again. Details: {exc}")

                if current_result and current_result.get("digest") == image_digest:
                    is_cat = current_result["label"] == "Cat"
                    emoji = "🐱" if is_cat else "🐶"
                    st.markdown(
                        f'<div class="result-card"><div class="result-animal">{emoji}</div><div class="result-label">PawVision predicts</div><div class="result-class">{current_result["label"]}</div><div class="metric-hint">Prediction confidence</div><div class="metric-value">{current_result["confidence"]:.1%}</div></div>',
                        unsafe_allow_html=True,
                    )
                    st.progress(current_result["confidence"], text=f"Confidence · {current_result['confidence']:.1%}")
                    st.markdown(f'<div class="probability-row"><span>🐶 Dog</span><b>{current_result["dog_probability"]:.1%}</b></div>', unsafe_allow_html=True)
                    st.progress(current_result["dog_probability"], text="Dog probability")
                    st.markdown(f'<div class="probability-row"><span>🐱 Cat</span><b>{current_result["cat_probability"]:.1%}</b></div>', unsafe_allow_html=True)
                    st.progress(current_result["cat_probability"], text="Cat probability")

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Clear image · Upload another", key="clear_upload"):
                st.session_state.pop("classification_result", None)
                st.session_state["uploader_version"] += 1
                st.rerun()
        except (UnidentifiedImageError, OSError, ValueError):
            st.error("This file isn’t a readable image. Please upload a valid JPG, JPEG, or PNG file.")

elif page == "Model Performance":
    st.markdown('<div class="eyebrow">Measured results</div>', unsafe_allow_html=True)
    st.title("How well did the model do?")
    st.markdown('<div class="soft-note">These figures come from the saved evaluation output for the held-out test split.</div>', unsafe_allow_html=True)
    required = ["test_accuracy", "test_loss", "precision_macro", "recall_macro", "f1_macro"]
    if not metrics or any(key not in metrics for key in required):
        st.warning("Evaluation metrics are incomplete. Run `python src/evaluate.py` to generate the performance results.")
    else:
        values = [
            ("Test accuracy", metric_value(metrics, "test_accuracy", percent=True), "Correct predictions on test images"),
            ("Test loss", metric_value(metrics, "test_loss", digits=4), "Lower is better"),
            ("Precision", metric_value(metrics, "precision_macro", percent=True), "How often predictions are right"),
            ("Recall", metric_value(metrics, "recall_macro", percent=True), "How many examples were found"),
            ("F1 score", metric_value(metrics, "f1_macro", percent=True), "Balance of precision and recall"),
        ]
        for start in (0, 3):
            cols = st.columns(min(3, len(values) - start), gap="medium")
            for col, (label, value, hint) in zip(cols, values[start:start + 3]):
                with col: metric_card(label, value, hint)
        st.markdown('<div class="section-kicker">What the model learned</div>', unsafe_allow_html=True)
        acc_path, loss_path = OUTPUTS_PATH / "accuracy.png", OUTPUTS_PATH / "loss.png"
        chart_cols = st.columns(2, gap="medium")
        with chart_cols[0]:
            if acc_path.is_file():
                st.image(str(acc_path), caption="Training and validation accuracy", width="stretch", alt="Training and validation accuracy chart")
            else:
                st.info("Training accuracy chart is not available yet.")
        with chart_cols[1]:
            if loss_path.is_file():
                st.image(str(loss_path), caption="Training and validation loss", width="stretch", alt="Training and validation loss chart")
            else:
                st.info("Training loss chart is not available yet.")

        confusion_path = OUTPUTS_PATH / "confusion_matrix.png"
        if confusion_path.is_file():
            st.markdown('<div class="section-kicker">Test set predictions</div>', unsafe_allow_html=True)
            st.image(str(confusion_path), caption="Confusion matrix for the test split", width=520, alt="Test set confusion matrix")
        report_path = OUTPUTS_PATH / "classification_report.txt"
        if report_path.is_file():
            st.markdown('<div class="section-kicker">Class-by-class detail</div>', unsafe_allow_html=True)
            with st.expander("View the Dog and Cat classification report", expanded=True):
                st.code(report_path.read_text(encoding="utf-8"), language="text")

elif page == "How It Works":
    st.markdown('<div class="eyebrow">A simple explanation</div>', unsafe_allow_html=True)
    st.title("From photo to prediction")
    st.markdown('<div class="soft-note">PawVision follows seven clear steps. Each one prepares the photo or helps the model make its decision.</div>', unsafe_allow_html=True)
    first_row = [
        (1, "Upload", "You choose a JPG, JPEG, or PNG photo."),
        (2, "Convert to RGB", "The image is put into the same three-colour format."),
        (3, "Resize", "The photo is resized to 128 × 128 pixels."),
        (4, "Normalize", "Pixel values are scaled to a range from 0 to 1."),
    ]
    cols = st.columns(4, gap="medium")
    for col, (number, title, copy) in zip(cols, first_row):
        with col: step_card(number, title, copy)
    st.markdown("<br>", unsafe_allow_html=True)
    second_row = [
        (5, "Analyze", "The CNN looks for visual patterns such as shapes, edges, and textures."),
        (6, "Choose a class", "The model compares its Dog and Cat probabilities."),
        (7, "Show confidence", "PawVision displays the selected class and its probability."),
    ]
    cols = st.columns(3, gap="medium")
    for col, (number, title, copy) in zip(cols, second_row):
        with col: step_card(number, title, copy)
    st.markdown('<div class="section-kicker">What does CNN mean?</div>', unsafe_allow_html=True)
    st.markdown('<div class="surface soft-note">A <b>convolutional neural network</b> is a computer model designed to learn patterns in images. PawVision’s CNN was trained from scratch using labeled dog and cat photos. During training it adjusted its internal values to reduce mistakes, then its performance was measured on a separate test set.</div>', unsafe_allow_html=True)

elif page == "About PawVision":
    st.markdown('<div class="eyebrow">The project</div>', unsafe_allow_html=True)
    st.title("Meet PawVision")
    st.markdown('<div class="soft-note">PawVision is a small image classification project that helps people explore how a computer can learn to tell cats from dogs.</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-kicker">The dataset</div>', unsafe_allow_html=True)
    source_counts = read_json(str(OUTPUTS_PATH / "dataset_summary.json")).get("counts", {})
    info_cols = st.columns(3, gap="medium")
    for col, (label, split) in zip(info_cols, [("Training", "train"), ("Validation", "validation"), ("Testing", "test")]):
        split_counts = source_counts.get(split, {})
        total = sum(split_counts.values()) if split_counts else dataset_counts.get(split, 0)
        with col: metric_card(f"{label} images", f"{total:,}", "Balanced between cats and dogs")
    st.markdown('<div class="section-kicker">Behind the scenes</div>', unsafe_allow_html=True)
    a, b = st.columns(2, gap="medium")
    with a:
        st.markdown('<div class="surface"><h3>Preparing the photo</h3><p class="soft-note">Each image is converted to RGB, resized to 128 × 128, and its pixel values are scaled to [0, 1]. Training photos also receive small random changes, such as a flip or slight rotation, so the model sees more variety.</p></div>', unsafe_allow_html=True)
    with b:
        st.markdown('<div class="surface"><h3>Learning image patterns</h3><p class="soft-note">The custom CNN uses several convolution layers to notice visual details, pooling layers to summarize them, and a final sigmoid output to estimate the probability that the image is a cat. The other probability is the dog score.</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-kicker">Model at a glance</div>', unsafe_allow_html=True)
    st.markdown('<div class="surface soft-note"><b>Custom CNN</b> · Three convolution blocks (32, 64, and 128 filters) · Batch normalization · ReLU · Max pooling and dropout · Global average pooling · Dense layer · Sigmoid output<br><br><b>Evaluation:</b> held-out test images, accuracy, precision, recall, F1 score, loss, confusion matrix, and class report.<br><br><a href="https://huggingface.co/datasets/shravya11/dog-cat-dataset" target="_blank">Hugging Face Dogs vs Cats dataset ↗</a></div>', unsafe_allow_html=True)

st.markdown('<div class="footer">PawVision · Dog and cat predictions are produced by the saved CNN and can be incorrect. Use clear images of one animal for the most useful demo.</div>', unsafe_allow_html=True)
