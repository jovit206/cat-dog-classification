"""PawVision: a friendly Streamlit interface for the PawVision dog-vs-cat classifier."""
from __future__ import annotations

import hashlib
import io
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"
MODEL_PATH = MODELS / "dogs_cats_model.keras"
GATE_PATH = MODELS / "pet_gate_imagenet.keras"
IMAGENET_LABELS_PATH = MODELS / "imagenet_class_index.json"
SCRATCH_METRICS_PATH = ROOT / "backups" / "scratch_cnn_20261119" / "outputs" / "metrics.json"
IMAGE_TYPES = ["jpg", "jpeg", "png", "webp", "bmp"]
# ImageNet class ranges: 151-268 are dog breeds, 281-285 are domestic cats.
DOG_CLASSES = np.arange(151, 269)
CAT_CLASSES = np.arange(281, 286)
PET_GATE_MIN = 0.20  # Below this ImageNet dog+cat mass, warn that the photo may not show a pet.
MAX_BATCH = 24

st.set_page_config(page_title="PawVision · Dog or Cat?", page_icon="🐾", layout="wide", initial_sidebar_state="collapsed")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');
    :root { --ink:#17243b; --muted:#5f6d84; --soft:#8b97a9; --blue:#476ee8; --blue-dark:#2f53c5; --teal:#2f9e8f; --amber:#c98a12; --line:#e4e9f2; --card:#ffffff; }
    html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
    .stApp { background:radial-gradient(ellipse at 90% 0%,#e7f1ff 0,transparent 40%),linear-gradient(135deg,#f7f9fd 0%,#f8f9fc 60%,#fff8f1 100%); color:var(--ink); }
    .block-container { padding-top:2.2rem; max-width:1180px; }
    h1,h2,h3 { font-family:'Manrope','DM Sans',sans-serif; color:var(--ink); letter-spacing:-.02em; }
    .hero { position:relative; overflow:hidden; padding:1.7rem 2rem; border-radius:22px; color:#fff; background:linear-gradient(112deg,#1d315a 0%,#3657a7 55%,#5797b7 100%); box-shadow:0 16px 40px #263f7024; margin-bottom:1.2rem; }
    .hero:after { content:'🐾'; position:absolute; right:3%; top:6%; font-size:6.5rem; opacity:.10; transform:rotate(-18deg); }
    .hero h1 { color:#fff; font-size:clamp(1.8rem,3.6vw,2.6rem); margin:0; padding:0; }
    .hero p { color:#e3ecff; font-size:1.02rem; max-width:640px; margin:.5rem 0 0; line-height:1.6; }
    .chips { display:flex; flex-wrap:wrap; gap:.45rem; margin-top:1rem; }
    .chip { background:#ffffff1f; border:1px solid #ffffff38; border-radius:99px; padding:.3rem .75rem; font-size:.8rem; color:#fff; }
    .eyebrow { color:#4e6fd2; text-transform:uppercase; letter-spacing:.12em; font-weight:700; font-size:.74rem; margin:1.1rem 0 .35rem; }
    .soft { color:var(--muted); font-size:.95rem; line-height:1.65; }
    .card { background:var(--card); border:1px solid var(--line); border-radius:18px; padding:1.1rem 1.2rem; box-shadow:0 6px 22px #273d680d; height:100%; }
    .metric-label { color:var(--muted); font-size:.82rem; font-weight:600; }
    .metric-value { color:#263d70; font:800 1.75rem 'Manrope',sans-serif; margin:.2rem 0 .05rem; }
    .metric-hint { color:var(--soft); font-size:.78rem; line-height:1.4; }
    .result { border-radius:20px; padding:1.4rem 1.4rem 1.2rem; border:1px solid #dce7fb; background:linear-gradient(145deg,#f5f8ff,#effaf8); text-align:center; }
    .result.warn { background:linear-gradient(145deg,#fffaf0,#fff4e0); border-color:#f1dcae; }
    .result-emoji { font-size:3.4rem; line-height:1.1; }
    .result-kicker { color:#61728d; font-size:.78rem; text-transform:uppercase; letter-spacing:.12em; font-weight:700; margin-top:.4rem; }
    .result-title { font:800 2.3rem 'Manrope',sans-serif; color:#25458c; margin:.05rem 0 .2rem; }
    .badge { display:inline-block; border-radius:99px; padding:.28rem .8rem; font-size:.82rem; font-weight:700; }
    .badge.high { background:#e3f6f1; color:#1e7a6c; } .badge.mid { background:#eaf0ff; color:#2f53c5; } .badge.low { background:#fff1d6; color:#93640a; }
    .split { display:flex; height:14px; border-radius:99px; overflow:hidden; margin:1.1rem 0 .4rem; background:#e9eef7; }
    .split .dog { background:linear-gradient(90deg,#f0a35b,#e98a3c); } .split .cat { background:linear-gradient(90deg,#5f86ee,#476ee8); }
    .split-legend { display:flex; justify-content:space-between; font-size:.88rem; color:#4f5d74; }
    .note { border-radius:14px; padding:.75rem .9rem; font-size:.9rem; line-height:1.5; margin-top:.8rem; text-align:left; }
    .note.warn { background:#fff6e3; border:1px solid #f1dcae; color:#6e4c07; }
    .note.info { background:#eef3ff; border:1px solid #d7e2fb; color:#2b3f6b; }
    .mini { background:#fff; border:1px solid var(--line); border-radius:14px; padding:.55rem .7rem .7rem; margin-top:.35rem; }
    .mini b { font-family:'Manrope',sans-serif; }
    .step { background:#fff; border:1px solid var(--line); border-radius:16px; padding:1rem; height:100%; }
    .step-num { display:grid; place-items:center; background:#edf2ff; color:#365dcf; border-radius:11px; width:34px; height:34px; font-weight:800; margin-bottom:.6rem; }
    .step-title { font-weight:700; color:#26334b; margin-bottom:.25rem; }
    .step-copy { color:var(--muted); font-size:.86rem; line-height:1.5; }
    .cm { display:grid; grid-template-columns:auto 1fr 1fr; gap:6px; max-width:420px; font-size:.88rem; }
    .cm div { padding:.7rem; border-radius:10px; text-align:center; }
    .cm .h { color:var(--muted); font-weight:600; background:transparent; }
    .cm .ok { background:#e3f6f1; color:#1e7a6c; font-weight:800; font-size:1.15rem; }
    .cm .bad { background:#fdecea; color:#a2392d; font-weight:800; font-size:1.15rem; }
    .footer { color:#9aa5b5; font-size:.8rem; border-top:1px solid var(--line); padding-top:1rem; margin-top:2.5rem; }
    div.stButton > button, div.stDownloadButton > button { border-radius:12px; font-weight:700; min-height:2.6rem; }
    div.stButton > button[kind="primary"] { background:linear-gradient(110deg,var(--blue),var(--blue-dark)); border:none; box-shadow:0 8px 18px #4367d52e; }
    [data-testid="stFileUploaderDropzone"] { border-radius:16px; border:1.5px dashed #b7c6e5; background:#fbfcff; padding:1.4rem; }
    .stTabs [data-baseweb="tab-list"] { gap:.3rem; }
    .stTabs [data-baseweb="tab"] { font-weight:600; }
    @media (max-width:700px) { .hero { padding:1.3rem; } .hero:after { font-size:4rem; } .result-title { font-size:1.9rem; } .metric-value { font-size:1.4rem; } }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------- data loading
def _stat_key(path: Path) -> tuple[int, int]:
    try:
        stat = path.stat()
        return stat.st_mtime_ns, stat.st_size
    except OSError:
        return 0, 0


@st.cache_data(show_spinner=False)
def _read_json_cached(path: str, _mtime: int, _size: int) -> dict:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def read_json(path: Path) -> dict:
    return _read_json_cached(str(path), *_stat_key(path))


@st.cache_resource(show_spinner=False)
def load_keras_model(path: str, _mtime: int, _size: int):
    import tensorflow as tf  # Imported lazily so the page renders before TensorFlow finishes loading.

    return tf.keras.models.load_model(path)


def get_classifier():
    return load_keras_model(str(MODEL_PATH), *_stat_key(MODEL_PATH))


def get_gate():
    if not GATE_PATH.is_file():
        return None
    try:
        return load_keras_model(str(GATE_PATH), *_stat_key(GATE_PATH))
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def imagenet_labels() -> list[str]:
    raw = read_json(IMAGENET_LABELS_PATH)
    return [raw.get(str(i), ["", f"class {i}"])[1].replace("_", " ") for i in range(1000)] if raw else []


metadata = read_json(MODELS / "model_metadata.json")
metrics = read_json(OUTPUTS / "metrics.json")
scratch_metrics = read_json(SCRATCH_METRICS_PATH)
dataset_summary = read_json(OUTPUTS / "dataset_summary.json")
THRESHOLD = float(metadata.get("validation_threshold", 0.5))
size = metadata.get("image_size", [128, 128])
IMAGE_SIZE = (int(size[0]), int(size[1])) if isinstance(size, list) and len(size) == 2 else (128, 128)
IS_TRANSFER = metadata.get("model_family") == "transfer_learning"
BACKBONE_NAME = {"mobilenetv2": "MobileNetV2", "efficientnetv2b0": "EfficientNetV2-B0"}.get(str(metadata.get("backbone", "")), "Custom CNN")
MODEL_LABEL = f"{BACKBONE_NAME} (pretrained)" if IS_TRANSFER else "Custom CNN"
EXAMPLE_ROOT = ROOT / ("dataset_224" if (ROOT / "dataset_224" / "test").is_dir() else "dataset") / "test"


def pct(value, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%" if isinstance(value, (int, float)) else "—"


# ----------------------------------------------------------------------------- inference
@dataclass
class Prediction:
    name: str
    digest: str
    image: Image.Image
    p_cat: float
    pet_score: float | None = None
    dog_mass: float | None = None
    cat_mass: float | None = None
    looks_like: str | None = None

    @property
    def label(self) -> str:
        return "Cat" if self.p_cat >= THRESHOLD else "Dog"

    @property
    def confidence(self) -> float:
        return self.p_cat if self.label == "Cat" else 1.0 - self.p_cat

    @property
    def emoji(self) -> str:
        return "🐱" if self.label == "Cat" else "🐶"

    @property
    def maybe_not_pet(self) -> bool:
        return self.pet_score is not None and self.pet_score < PET_GATE_MIN

    @property
    def maybe_both(self) -> bool:
        return self.dog_mass is not None and self.cat_mass is not None and min(self.dog_mass, self.cat_mass) >= 0.2

    @property
    def certainty(self) -> tuple[str, str]:
        if self.confidence >= 0.9:
            return "Very confident", "high"
        if self.confidence >= 0.7:
            return "Fairly confident", "mid"
        return "Not sure", "low"


def open_image(data: bytes) -> Image.Image:
    with Image.open(io.BytesIO(data)) as opened:
        opened.load()
        return ImageOps.exif_transpose(opened).convert("RGB")


def to_batch(images: list[Image.Image], size: tuple[int, int]) -> np.ndarray:
    # LANCZOS matches how the training images were resized in preprocessing.py.
    return np.stack([np.asarray(img.resize(size, Image.Resampling.LANCZOS), dtype=np.float32) / 255.0 for img in images])


def classify(items: list[tuple[str, bytes]]) -> tuple[list[Prediction], list[str]]:
    """Classify every readable image in one batch; return predictions and names that failed."""
    cache: dict[str, Prediction] = st.session_state.setdefault("prediction_cache", {})
    model_key = f"{_stat_key(MODEL_PATH)}:{THRESHOLD}"
    if st.session_state.get("prediction_cache_model") != model_key:
        cache.clear()
        st.session_state["prediction_cache_model"] = model_key
    results: list[Prediction] = []
    failed: list[str] = []
    pending: list[Prediction] = []
    for name, data in items:
        digest = hashlib.sha256(data).hexdigest()
        if digest in cache:
            results.append(cache[digest])
            continue
        try:
            image = open_image(data)
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
            failed.append(name)
            continue
        prediction = Prediction(name=name, digest=digest, image=image, p_cat=0.5)
        pending.append(prediction)
        results.append(prediction)

    if pending:
        images = [p.image for p in pending]
        p_cat = np.asarray(get_classifier().predict(to_batch(images, IMAGE_SIZE), verbose=0)).reshape(-1)
        gate = get_gate()
        gate_probs = None
        if gate is not None:
            gate_input = to_batch(images, (224, 224)) * 2.0 - 1.0  # MobileNetV2 expects [-1, 1].
            gate_probs = np.asarray(gate.predict(gate_input, verbose=0))
        labels = imagenet_labels()
        for index, prediction in enumerate(pending):
            prediction.p_cat = float(np.clip(np.nan_to_num(p_cat[index], nan=0.5), 0.0, 1.0))
            if gate_probs is not None:
                probs = gate_probs[index]
                prediction.dog_mass = float(probs[DOG_CLASSES].sum())
                prediction.cat_mass = float(probs[CAT_CLASSES].sum())
                prediction.pet_score = prediction.dog_mass + prediction.cat_mass
                if labels:
                    prediction.looks_like = labels[int(np.argmax(probs))]
            cache[prediction.digest] = prediction
    return results, failed


def attention_overlay(image: Image.Image) -> Image.Image | None:
    """Class activation map: which regions pushed the model toward its answer."""
    model = get_classifier()
    try:
        preprocess = model.get_layer("preprocess")
        backbone = model.get_layer("backbone")
        head = model.get_layer("cat_probability")
    except ValueError:
        return None
    import matplotlib

    batch = to_batch([image], IMAGE_SIZE)
    features = np.asarray(backbone(preprocess(batch), training=False))[0]  # (h, w, channels)
    weights = np.asarray(head.get_weights()[0]).reshape(-1)
    cam = features @ weights
    p_cat = float(np.asarray(model(batch, training=False)).reshape(-1)[0])
    if p_cat < THRESHOLD:
        cam = -cam  # Show evidence for "Dog" when the answer is Dog.
    cam = np.maximum(cam, 0)
    if cam.max() <= 0:
        return None
    cam = cam / cam.max()
    heat = Image.fromarray(np.uint8(cam * 255)).resize(image.size, Image.Resampling.BICUBIC)
    heat_array = np.asarray(heat, dtype=np.float32) / 255.0
    colored = matplotlib.colormaps["turbo"](heat_array)[..., :3]
    base = np.asarray(image, dtype=np.float32) / 255.0
    alpha = 0.55 * heat_array[..., None]
    blended = base * (1 - alpha) + colored * alpha
    return Image.fromarray(np.uint8(np.clip(blended, 0, 1) * 255))


# ----------------------------------------------------------------------------- UI pieces
def metric_card(label: str, value: str, hint: str = "") -> None:
    st.markdown(f'<div class="card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-hint">{hint}</div></div>', unsafe_allow_html=True)


def step_card(number: int, title: str, copy: str) -> None:
    st.markdown(f'<div class="step"><div class="step-num">{number}</div><div class="step-title">{title}</div><div class="step-copy">{copy}</div></div>', unsafe_allow_html=True)


def result_card(pred: Prediction) -> None:
    certainty, tone = pred.certainty
    dog_pct, cat_pct = (1 - pred.p_cat) * 100, pred.p_cat * 100
    warn = pred.maybe_not_pet
    title = f"Probably a {pred.label.lower()}" if tone == "low" or warn else f"It's a {pred.label.lower()}!"
    notes = ""
    if warn:
        guess = f" It looks more like <b>{pred.looks_like}</b> to a general image model." if pred.looks_like else ""
        notes += f'<div class="note warn">🤔 <b>This might not be a dog or a cat.</b>{guess} PawVision only knows these two animals, so it has to pick one of them anyway.</div>'
    elif pred.maybe_both:
        notes += '<div class="note info">👀 There may be <b>both a dog and a cat</b> in this photo. PawVision picks just one, so try a photo with a single animal.</div>'
    elif tone == "low":
        notes += '<div class="note warn">The model is unsure about this one. A clearer, closer photo of one animal usually helps.</div>'
    st.markdown(
        f'<div class="result{" warn" if warn else ""}">'
        f'<div class="result-emoji">{pred.emoji}</div>'
        f'<div class="result-kicker">PawVision thinks</div>'
        f'<div class="result-title">{title}</div>'
        f'<span class="badge {tone}">{certainty} · {pred.confidence:.0%}</span>'
        f'<div class="split" role="img" aria-label="Dog {dog_pct:.0f} percent, cat {cat_pct:.0f} percent">'
        f'<div class="dog" style="width:{dog_pct:.2f}%"></div><div class="cat" style="width:{cat_pct:.2f}%"></div></div>'
        f'<div class="split-legend"><span>🐶 Dog <b>{dog_pct:.1f}%</b></span><span><b>{cat_pct:.1f}%</b> Cat 🐱</span></div>'
        f'{notes}</div>',
        unsafe_allow_html=True,
    )


def footer() -> None:
    st.markdown(
        f'<div class="footer">PawVision · {MODEL_LABEL} · Your photos are processed in memory on this computer and never saved. Predictions can be wrong, so treat them as a demo, not a verdict.</div>',
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------- input handling
def _set_source(source: str) -> None:
    st.session_state["source"] = source


def _pick_example(path: str) -> None:
    st.session_state["source"] = "example"
    st.session_state["example_path"] = path


def _reset_inputs() -> None:
    st.session_state["source"] = None
    st.session_state["example_path"] = None
    st.session_state["input_version"] = st.session_state.get("input_version", 0) + 1


def _shuffle_examples() -> None:
    st.session_state["example_seed"] = random.randrange(1_000_000)


@st.cache_data(show_spinner=False)
def example_paths(seed: int, per_class: int = 4) -> list[str]:
    rng = random.Random(seed)
    chosen: list[str] = []
    for label in ("Cat", "Dog"):
        files = sorted((EXAMPLE_ROOT / label).glob("*.jpg"))
        chosen += [str(p) for p in rng.sample(files, min(per_class, len(files)))]
    rng.shuffle(chosen)
    return chosen


def collect_inputs() -> list[tuple[str, bytes]]:
    version = st.session_state.setdefault("input_version", 0)
    st.session_state.setdefault("source", None)
    st.session_state.setdefault("example_seed", 7)

    upload_tab, example_tab, camera_tab = st.tabs(["📁  Upload photos", "🖼️  Try an example", "📷  Use camera"])
    with upload_tab:
        uploads = st.file_uploader(
            "Drop one or more photos here",
            type=IMAGE_TYPES,
            accept_multiple_files=True,
            key=f"upload_{version}",
            on_change=_set_source,
            args=("upload",),
            help="JPG, PNG, WEBP or BMP. You can select several photos at once.",
        )
        st.caption("Tip: a clear photo with one animal filling most of the frame works best.")
    with example_tab:
        st.caption("No photo handy? Pick one of these. The model never saw them during training.")
        paths = example_paths(st.session_state["example_seed"])
        cols = st.columns(4, gap="small")
        for index, path in enumerate(paths):
            with cols[index % 4]:
                st.image(path, width="stretch")
                st.button("Try this", key=f"ex_{index}_{Path(path).stem}", on_click=_pick_example, args=(path,), width="stretch")
        st.button("🔀  Show different examples", on_click=_shuffle_examples)
    with camera_tab:
        enable = st.toggle("Turn on camera", key=f"camera_on_{version}", help="Your browser will ask for permission. Nothing is recorded or saved.")
        shot = st.camera_input("Take a photo of your pet", key=f"camera_{version}", on_change=_set_source, args=("camera",), label_visibility="collapsed") if enable else None

    source = st.session_state["source"]
    if source == "upload" and uploads:
        if len(uploads) > MAX_BATCH:
            st.info(f"Showing the first {MAX_BATCH} of {len(uploads)} photos.")
        return [(u.name, u.getvalue()) for u in uploads[:MAX_BATCH]]
    if source == "camera" and shot is not None:
        return [("Camera photo", shot.getvalue())]
    if source == "example" and st.session_state.get("example_path"):
        path = Path(st.session_state["example_path"])
        if path.is_file():
            return [(f"Example: {path.parent.name.lower()} photo", path.read_bytes())]
    # Fall back to whatever is still present (e.g. a rerun after a widget changed).
    if uploads:
        return [(u.name, u.getvalue()) for u in uploads[:MAX_BATCH]]
    if shot is not None:
        return [("Camera photo", shot.getvalue())]
    return []


# ----------------------------------------------------------------------------- pages
def page_classify() -> None:
    acc = metrics.get("test_accuracy")
    n_test = metrics.get("test_images")
    chips = [f"🎯 {pct(acc)} accurate on {n_test:,} unseen photos" if isinstance(acc, float) and n_test else None, f"🧠 {MODEL_LABEL}", "🔒 Photos stay on this computer"]
    st.markdown(
        '<div class="hero"><h1>Dog or cat? Let\'s find out.</h1>'
        "<p>Add a photo and PawVision will tell you what it sees, how sure it is, and which parts of the picture it looked at.</p>"
        f'<div class="chips">{"".join(f"<span class=chip>{c}</span>" for c in chips if c)}</div></div>',
        unsafe_allow_html=True,
    )
    if not MODEL_PATH.is_file():
        st.error("The trained model file is missing, so predictions are turned off. Restore `models/dogs_cats_model.keras` and reload the page.")
        footer()
        return

    items = collect_inputs()
    if not items:
        st.markdown('<div class="note info" style="margin-top:1.2rem">👆 Upload a photo, pick an example, or use your camera. The answer appears right away, with no extra button to press.</div>', unsafe_allow_html=True)
        footer()
        return

    try:
        with st.spinner("Looking at your photo…" if len(items) == 1 else f"Looking at {len(items)} photos…"):
            predictions, failed = classify(items)
    except Exception as exc:  # A broken model file should not crash the whole page.
        st.error(f"PawVision couldn't run the model. Check that the model file is intact, then try again. ({exc})")
        footer()
        return
    for name in failed:
        st.warning(f"**{name}** couldn't be opened as an image. Please try a JPG or PNG file.")
    if not predictions:
        footer()
        return

    st.markdown('<div class="eyebrow">Result</div>' if len(predictions) == 1 else f'<div class="eyebrow">Results · {len(predictions)} photos</div>', unsafe_allow_html=True)
    if len(predictions) == 1:
        show_single(predictions[0])
    else:
        show_batch(predictions)
    st.button("↺  Start over with a new photo", on_click=_reset_inputs)
    footer()


def show_single(pred: Prediction) -> None:
    left, right = st.columns([1.1, 1], gap="large")
    with right:
        result_card(pred)
        with st.expander("What do these numbers mean?"):
            st.markdown(
                f"- The bar shows how the model splits its belief between **dog** and **cat**. The two sides always add up to 100%.\n"
                f"- **Very confident** means 90% or more, **fairly confident** 70–90%, and **not sure** below 70%.\n"
                f"- Confidence is about *this photo*. The model's overall track record is a separate number: {pct(metrics.get('test_accuracy'))} correct on photos it had never seen.\n"
                + (f"- It answers *cat* when the cat score is at least {THRESHOLD:.0%}. That cut-off was tuned on validation photos." if abs(THRESHOLD - 0.5) >= 0.005 else "")
            )
    with left:
        can_explain = IS_TRANSFER
        show_heat = st.toggle("🔍 Show what the model focused on", value=False, disabled=not can_explain, help="Highlights the parts of the photo that pushed the model toward its answer. Warm colours mean more influence." if can_explain else "Only available for the pretrained model.")
        shown = pred.image
        caption = pred.name
        if show_heat:
            with st.spinner("Drawing the attention map…"):
                overlay = attention_overlay(pred.image)
            if overlay is not None:
                shown, caption = overlay, "Warm areas (red/yellow) influenced the answer most"
            else:
                st.caption("No clear focus area for this photo.")
        st.image(shown, caption=caption, width="stretch")


def show_batch(predictions: list[Prediction]) -> None:
    cats = sum(p.label == "Cat" for p in predictions)
    dogs = len(predictions) - cats
    unsure = sum(p.certainty[1] == "low" or p.maybe_not_pet for p in predictions)
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1: metric_card("Cats", f"🐱 {cats}")
    with c2: metric_card("Dogs", f"🐶 {dogs}")
    with c3: metric_card("Need a second look", f"🤔 {unsure}", "Low confidence or maybe not a pet")
    st.write("")
    cols = st.columns(4, gap="medium")
    for index, pred in enumerate(predictions):
        certainty, tone = pred.certainty
        flag = " · ⚠️ maybe not a pet" if pred.maybe_not_pet else ""
        with cols[index % 4]:
            st.image(pred.image, width="stretch")
            st.markdown(f'<div class="mini">{pred.emoji} <b>{pred.label}</b> <span class="badge {tone}" style="float:right">{pred.confidence:.0%}</span><div class="metric-hint" style="margin-top:.3rem">{pred.name[:34]}{flag}</div></div>', unsafe_allow_html=True)
    table = pd.DataFrame([{"photo": p.name, "prediction": p.label, "confidence": round(p.confidence, 4), "cat_probability": round(p.p_cat, 4), "certainty": p.certainty[0], "maybe_not_a_pet": p.maybe_not_pet} for p in predictions])
    st.download_button("⬇️  Download results (CSV)", table.to_csv(index=False).encode("utf-8"), file_name="pawvision_results.csv", mime="text/csv")


def page_performance() -> None:
    st.markdown('<div class="eyebrow">Measured results</div>', unsafe_allow_html=True)
    st.title("How good is PawVision?")
    if not metrics.get("test_accuracy"):
        st.warning("No evaluation results yet. Run `python src/evaluate.py` to generate them.")
        footer()
        return
    cm = metrics.get("confusion_matrix", [[0, 0], [0, 0]])
    n = metrics.get("test_images", sum(map(sum, cm)))
    correct = cm[0][0] + cm[1][1]
    st.markdown(f'<div class="soft">We kept <b>{n:,} photos</b> aside that the model never saw while it was learning. It got <b>{correct:,} of them right</b>, which is <b>{pct(metrics["test_accuracy"])}</b>.</div>', unsafe_allow_html=True)

    cols = st.columns(4, gap="medium")
    cards = [
        ("Accuracy", pct(metrics.get("test_accuracy")), "Share of test photos it got right"),
        ("Cat recall", pct(metrics.get("cat_recall")), "Cats it correctly called cats"),
        ("Dog recall", pct(metrics.get("dog_recall")), "Dogs it correctly called dogs"),
        ("ROC AUC", f"{metrics.get('roc_auc', 0):.3f}", "1.0 = perfect ranking of cats above dogs"),
    ]
    for col, card in zip(cols, cards):
        with col: metric_card(*card)

    if scratch_metrics.get("test_accuracy") and IS_TRANSFER:
        st.markdown('<div class="eyebrow">Before and after</div>', unsafe_allow_html=True)
        old, new = scratch_metrics["test_accuracy"], metrics["test_accuracy"]
        old_err, new_err = 1 - old, 1 - new
        st.markdown(
            f'<div class="soft">The first version was a small network trained from scratch on 3,500 photos. Switching to a {BACKBONE_NAME} pretrained on ImageNet took accuracy from <b>{pct(old)}</b> to <b>{pct(new)}</b>, '
            f"so it now makes about <b>{old_err / max(new_err, 1e-9):.0f}× fewer mistakes</b> on the same test photos.</div>",
            unsafe_allow_html=True,
        )
        compare = pd.DataFrame(
            {"Scratch CNN": [scratch_metrics.get(k, np.nan) * 100 for k in ("test_accuracy", "f1_macro", "cat_recall", "dog_recall")],
             MODEL_LABEL: [metrics.get(k, np.nan) * 100 for k in ("test_accuracy", "f1_macro", "cat_recall", "dog_recall")]},
            index=["Accuracy", "Macro F1", "Cat recall", "Dog recall"],
        )
        st.bar_chart(compare, stack=False, height=280, y_label="%", color=["#b7c3d9", "#476ee8"])

    st.markdown('<div class="eyebrow">Where the right and wrong answers are</div>', unsafe_allow_html=True)
    left, right = st.columns([1, 1.2], gap="large")
    with left:
        st.markdown(
            '<div class="cm"><div class="h"></div><div class="h">Said “Dog”</div><div class="h">Said “Cat”</div>'
            f'<div class="h">Real dog</div><div class="ok">{cm[0][0]}</div><div class="bad">{cm[0][1]}</div>'
            f'<div class="h">Real cat</div><div class="bad">{cm[1][0]}</div><div class="ok">{cm[1][1]}</div></div>',
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(f'<div class="soft">Green squares are correct answers. Red squares are mistakes: <b>{cm[0][1]}</b> dogs were called cats and <b>{cm[1][0]}</b> cats were called dogs.</div>', unsafe_allow_html=True)

    mistakes = [
        (OUTPUTS / "error_analysis" / "final_test" / f"{name}.png", f"{title} ({count})", count)
        for name, title, count in (("false_negative_cats", "Cats it called dogs", cm[1][0]), ("false_positive_cats", "Dogs it called cats", cm[0][1]))
    ]
    if any(path.is_file() for path, _, _ in mistakes):
        st.markdown('<div class="eyebrow">Mistakes it made</div>', unsafe_allow_html=True)
        tabs = st.tabs([title for _, title, _ in mistakes])
        for tab, (path, title, count) in zip(tabs, mistakes):
            with tab:
                if count == 0:
                    st.success("None in the test set! 🎉")
                elif path.is_file():
                    st.image(str(path), width="stretch")
                else:
                    st.caption("No examples saved.")

    with st.expander("📈  Technical charts and full report"):
        charts = [("accuracy.png", "Training & validation accuracy"), ("loss.png", "Training & validation loss"), ("roc_curve.png", "ROC curve"), ("precision_recall_curve.png", "Precision–recall curve")]
        for row in range(0, len(charts), 2):
            c = st.columns(2, gap="medium")
            for col, (file, caption) in zip(c, charts[row:row + 2]):
                with col:
                    if (OUTPUTS / file).is_file():
                        st.image(str(OUTPUTS / file), caption=caption, width="stretch")
        extra = [("Loss", f"{metrics.get('test_loss', 0):.4f}"), ("Macro precision", pct(metrics.get("precision_macro"), 2)), ("Macro recall", pct(metrics.get("recall_macro"), 2)), ("Macro F1", pct(metrics.get("f1_macro"), 2)), ("Average precision", f"{metrics.get('average_precision', 0):.4f}"), ("Decision threshold", f"{metrics.get('decision_threshold', THRESHOLD):.2f}")]
        st.dataframe(pd.DataFrame(extra, columns=["Metric", "Value"]), hide_index=True, width="stretch")
        report = OUTPUTS / "classification_report.txt"
        if report.is_file():
            st.code(report.read_text(encoding="utf-8"), language="text")
    footer()


def page_how() -> None:
    st.markdown('<div class="eyebrow">A plain-language tour</div>', unsafe_allow_html=True)
    st.title("How PawVision works")
    w, h = IMAGE_SIZE
    steps = [
        (1, "You add a photo", "Upload a photo, take one with your camera, or try an example."),
        (2, "It's tidied up", f"PawVision turns the photo the right way up, converts it to colour (RGB) and resizes it to {w} × {h} pixels."),
        (3, "The model looks", "A neural network finds shapes like ears, snouts, whiskers and fur, and weighs the evidence."),
        (4, "You get an answer", "Dog or cat, how sure it is, and a heatmap showing which parts of the photo mattered."),
    ]
    cols = st.columns(4, gap="medium")
    for col, step in zip(cols, steps):
        with col: step_card(*step)

    st.markdown('<div class="eyebrow">Why it\'s accurate</div>', unsafe_allow_html=True)
    a, b = st.columns(2, gap="medium")
    with a:
        st.markdown(
            f'<div class="card"><h3>Learning from a head start</h3><p class="soft">PawVision uses <b>transfer learning</b>. Its {BACKBONE_NAME} backbone first learned to recognise everyday objects from over a million ImageNet photos. '
            "We then taught it the dog-versus-cat task with our own photos: first a new final layer, then gentle fine-tuning of the top of the network.</p></div>"
            if IS_TRANSFER else
            '<div class="card"><h3>A custom CNN</h3><p class="soft">This model is a convolutional neural network trained from scratch on labelled dog and cat photos.</p></div>',
            unsafe_allow_html=True,
        )
    with b:
        st.markdown(
            '<div class="card"><h3>Fair testing</h3><p class="soft">The photos were split into three groups that never overlap: <b>training</b> (to learn), <b>validation</b> (to choose the best model and its cat/dog cut-off) and <b>test</b> (checked only at the end). '
            "So the accuracy you see reflects photos the model had never seen.</p></div>",
            unsafe_allow_html=True,
        )

    st.markdown('<div class="eyebrow">The dataset</div>', unsafe_allow_html=True)
    counts = dataset_summary.get("counts", {})
    cols = st.columns(3, gap="medium")
    for col, (label, split, hint) in zip(cols, [("Training", "train", "Used to learn"), ("Validation", "validation", "Used to pick the best model"), ("Test", "test", "Used once, for the final score")]):
        total = sum(counts.get(split, {}).values())
        with col: metric_card(f"{label} photos", f"{total:,}" if total else "—", f"{hint} · half cats, half dogs")
    st.caption("Source: the [shravya11/dog-cat-dataset](https://huggingface.co/datasets/shravya11/dog-cat-dataset) on Hugging Face.")

    st.markdown('<div class="eyebrow">Good to know</div>', unsafe_allow_html=True)
    st.markdown(
        "- **It only knows two answers.** Show it a rabbit and it will still say dog or cat. PawVision tries to warn you when a photo doesn't look like a pet.\n"
        "- **One animal per photo** works best. With both in the frame it can only pick one.\n"
        "- **Privacy:** photos are processed in memory on this computer and aren't stored or sent anywhere.\n"
        f"- **Model details:** {MODEL_LABEL}, {w}×{h} input, cat/dog cut-off {THRESHOLD:.2f}"
        + (f", selected experiment `{metadata['selected_experiment']}`." if metadata.get("selected_experiment") else ".")
    )
    footer()


nav = st.navigation(
    [
        st.Page(page_classify, title="Classify", icon="🐾", url_path="classify", default=True),
        st.Page(page_performance, title="Model results", icon="📊", url_path="results"),
        st.Page(page_how, title="How it works", icon="💡", url_path="how-it-works"),
    ],
    position="top",
)
nav.run()
