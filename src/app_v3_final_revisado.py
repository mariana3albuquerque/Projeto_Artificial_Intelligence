from __future__ import annotations

import os
import warnings

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
warnings.filterwarnings("ignore", message=".*The structure of `inputs` doesn.*")
warnings.filterwarnings("ignore", category=UserWarning, module="keras.*")

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageDraw
from tensorflow import keras


# ============================================================
# Custom loss used by the v3 model
# ============================================================

class SparseCategoricalFocalLoss(keras.losses.Loss):
    """
    Multiclass focal loss for one-hot labels and softmax outputs.
    Needed only to reload the v3 .keras model, which was trained with this loss.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        label_smoothing: float = 0.01,
        name: str = "focal_loss",
    ) -> None:
        super().__init__(name=name)
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        n_classes = tf.cast(tf.shape(y_pred)[-1], tf.float32)

        if self.label_smoothing > 0:
            y_true = y_true * (1.0 - self.label_smoothing) + self.label_smoothing / n_classes

        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce = -y_true * tf.math.log(y_pred)
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)
        focal_weight = tf.pow(1.0 - p_t, self.gamma)
        loss = focal_weight * ce
        return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))

    def get_config(self) -> dict:
        base = super().get_config()
        base.update({"gamma": self.gamma, "label_smoothing": self.label_smoothing})
        return base


# ============================================================
# Main configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL_PATH = (
    PROJECT_ROOT / "reports" / "model_v3_mel_focal" / "cnn_ham10000_v3.keras"
)
DEFAULT_POLICY_PATH = (
    PROJECT_ROOT / "reports" / "model_v3_mel_focal" / "decision_policy_v3.json"
)
DEFAULT_METRICS_PATH = (
    PROJECT_ROOT / "reports" / "model_v3_mel_focal" / "metrics_v3.json"
)

IMAGE_SIZE = (224, 224)

CLASS_NAMES: List[str] = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
MEL_INDEX = CLASS_NAMES.index("mel")

CLASS_DESCRIPTIONS: Dict[str, str] = {
    "akiec": "Actinic keratoses / intraepithelial carcinoma",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions",
}

CLASS_DESCRIPTIONS_PT: Dict[str, str] = {
    "akiec": "Queratose actínica / carcinoma intraepitelial",
    "bcc": "Carcinoma basocelular",
    "bkl": "Lesão benigna semelhante à queratose",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Nevo melanocítico",
    "vasc": "Lesão vascular",
}

DEFAULT_MELANOMA_THRESHOLD = 0.23
DEFAULT_INTERMEDIATE_THRESHOLD = 0.15
DEFAULT_BBOX_THRESHOLD = 0.60


# ============================================================
# Style
# ============================================================

st.set_page_config(
    page_title="Skin Lesion Triage Support v3",
    page_icon="🩺",
    layout="wide",
)

st.markdown(
    """
<style>
    .main-title {
        font-size: 2.15rem;
        font-weight: 800;
        margin-bottom: 0.15rem;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #A7B0C0;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background: #111827;
        border: 1px solid #283548;
        border-radius: 16px;
        padding: 18px 20px;
        min-height: 120px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.18);
    }
    .metric-label {
        color: #A7B0C0;
        font-size: 0.82rem;
        font-weight: 650;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.45rem;
    }
    .metric-value {
        color: #F9FAFB;
        font-size: 1.45rem;
        font-weight: 800;
        line-height: 1.25;
    }
    .risk-high {
        background: #3A151D;
        border-left: 7px solid #EF4444;
        padding: 15px 18px;
        border-radius: 12px;
        color: #FECACA;
    }
    .risk-mid {
        background: #3A3411;
        border-left: 7px solid #F59E0B;
        padding: 15px 18px;
        border-radius: 12px;
        color: #FEF3C7;
    }
    .risk-low {
        background: #11351F;
        border-left: 7px solid #22C55E;
        padding: 15px 18px;
        border-radius: 12px;
        color: #BBF7D0;
    }
    .method-note {
        background: #0F172A;
        border: 1px solid #334155;
        padding: 12px 14px;
        border-radius: 12px;
        color: #CBD5E1;
    }
    .small-muted {
        color: #A7B0C0;
        font-size: 0.9rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# Generic helpers
# ============================================================

@st.cache_resource(show_spinner=False)
def load_trained_model(model_path: str) -> keras.Model:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Modelo não encontrado em: {path}")

    return keras.models.load_model(
        path,
        custom_objects={"SparseCategoricalFocalLoss": SparseCategoricalFocalLoss},
        compile=False,
    )


@st.cache_data(show_spinner=False)
def load_json_file(path_text: str) -> Dict:
    path = Path(path_text)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_decision_policy(policy_path: Path) -> Dict[str, float]:
    policy = load_json_file(str(policy_path))
    if "melanoma_threshold" not in policy:
        policy["melanoma_threshold"] = DEFAULT_MELANOMA_THRESHOLD
    return policy


def format_percentage(value: float) -> str:
    return f"{value * 100:.2f}%"


def pil_to_rgb_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"))


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    The v3 model architecture already contains EfficientNet preprocess_input.
    Therefore, the app sends RGB float32 pixels in the 0-255 range.
    """
    image = image.convert("RGB").resize(IMAGE_SIZE)
    array = np.asarray(image).astype(np.float32)
    return np.expand_dims(array, axis=0)




def call_model_for_gradients(model: keras.Model, input_tensor: tf.Tensor) -> tf.Tensor:
    """
    Calls a Keras model in a way that is compatible with Functional models
    saved with named inputs. This avoids most harmless input-structure warnings
    during saliency/Grad-CAM generation.
    """
    try:
        return model(input_tensor, training=False)
    except Exception:
        input_names = getattr(model, "input_names", None)
        if input_names:
            return model({input_names[0]: input_tensor}, training=False)
        raise

def predict_probabilities(model: keras.Model, image: Image.Image) -> Dict[str, float]:
    input_array = preprocess_image(image)
    probabilities = model.predict(input_array, verbose=0)[0]
    return {name: float(prob) for name, prob in zip(CLASS_NAMES, probabilities)}


def apply_melanoma_policy(
    probs_by_class: Dict[str, float],
    melanoma_threshold: float,
) -> Tuple[str, float, str, float]:
    argmax_class = max(probs_by_class, key=probs_by_class.get)
    argmax_prob = probs_by_class[argmax_class]
    melanoma_prob = probs_by_class["mel"]

    if melanoma_prob >= melanoma_threshold:
        return "mel", melanoma_prob, argmax_class, argmax_prob
    return argmax_class, argmax_prob, argmax_class, argmax_prob


def get_melanoma_risk_level(
    melanoma_prob: float,
    melanoma_threshold: float,
    intermediate_threshold: float,
) -> str:
    if melanoma_prob >= melanoma_threshold:
        return "high"
    if melanoma_prob >= intermediate_threshold:
        return "intermediate"
    return "low"


def risk_label(risk_level: str) -> str:
    if risk_level == "high":
        return "Alto risco / suspeito para melanoma"
    if risk_level == "intermediate":
        return "Risco intermediário"
    return "Baixa probabilidade pelo modelo"


def triage_message(
    risk_level: str,
    final_class: str,
) -> str:
    if risk_level == "high":
        return (
            "A imagem foi sinalizada como suspeita para melanoma pela regra melanoma-sensitive. "
            "No contexto de triagem, recomenda-se avaliação médica especializada."
        )
    if risk_level == "intermediate":
        return (
            "A probabilidade de melanoma ficou em uma faixa intermediária. O sistema não alterou "
            "automaticamente a classe final para melanoma, mas recomenda cautela clínica, especialmente "
            "se houver assimetria, bordas irregulares, variação de cor, crescimento ou sintomas."
        )
    return (
        f"A classe final foi {final_class}. A probabilidade de melanoma ficou abaixo do limiar intermediário. "
        "Mesmo assim, o resultado não substitui avaliação médica."
    )


# ============================================================
# Image quality
# ============================================================

def compute_image_quality(image: Image.Image) -> Dict[str, float]:
    rgb = pil_to_rgb_array(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return {
        "brightness": float(np.mean(gray)),
        "contrast": float(np.std(gray)),
        "blur_score": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
    }


def quality_status(image: Image.Image) -> Tuple[Dict[str, float], List[str]]:
    quality = compute_image_quality(image)
    warnings: List[str] = []

    if quality["brightness"] < 45:
        warnings.append("imagem escura")
    elif quality["brightness"] > 220:
        warnings.append("imagem muito clara/superexposta")

    if quality["contrast"] < 20:
        warnings.append("baixo contraste")

    if quality["blur_score"] < 25:
        warnings.append("possível desfoque")

    return quality, warnings


def render_quality_section(image: Image.Image) -> None:
    quality, warnings = quality_status(image)
    with st.expander("Checklist de qualidade da imagem", expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.metric("Brilho", f"{quality['brightness']:.1f}")
        c2.metric("Contraste", f"{quality['contrast']:.1f}")
        c3.metric("Nitidez", f"{quality['blur_score']:.1f}")

        if warnings:
            st.warning(
                "Possíveis problemas: " + ", ".join(warnings) + 
                ". Esses fatores podem reduzir a confiabilidade da predição."
            )
        else:
            st.success("A imagem passou nos indicadores básicos de qualidade.")

        st.caption(
            "Recomendação: foto próxima, bem iluminada, em foco, com a lesão centralizada e pouca região de fundo."
        )


# ============================================================
# Cropping and approximate segmentation
# ============================================================

def crop_image_by_box(image: Image.Image, box: Tuple[int, int, int, int]) -> Image.Image:
    x_min, y_min, x_max, y_max = box
    x_min = max(0, int(x_min))
    y_min = max(0, int(y_min))
    x_max = min(image.width, int(x_max))
    y_max = min(image.height, int(y_max))
    if x_max <= x_min or y_max <= y_min:
        return image
    return image.crop((x_min, y_min, x_max, y_max))


def expand_box(
    box: Tuple[int, int, int, int],
    image_size: Tuple[int, int],
    padding_ratio: float = 0.12,
) -> Tuple[int, int, int, int]:
    width, height = image_size
    x_min, y_min, x_max, y_max = box
    bw = max(1, x_max - x_min)
    bh = max(1, y_max - y_min)
    pad_x = int(bw * padding_ratio)
    pad_y = int(bh * padding_ratio)
    return (
        max(0, x_min - pad_x),
        max(0, y_min - pad_y),
        min(width, x_max + pad_x),
        min(height, y_max + pad_y),
    )


def draw_box_on_image(
    image: Image.Image,
    box: Optional[Tuple[int, int, int, int]],
    label: str,
    color: str = "red",
) -> Image.Image:
    image = image.convert("RGB").copy()
    if box is None:
        return image
    draw = ImageDraw.Draw(image)
    x_min, y_min, x_max, y_max = box
    draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=5)
    draw.text((x_min, max(0, y_min - 24)), label, fill=color)
    return image


def estimate_background_lab(rgb: np.ndarray, corner_fraction: float = 0.10) -> np.ndarray:
    h, w, _ = rgb.shape
    ch = max(5, int(h * corner_fraction))
    cw = max(5, int(w * corner_fraction))
    corners = [rgb[:ch, :cw], rgb[:ch, -cw:], rgb[-ch:, :cw], rgb[-ch:, -cw:]]
    corner_pixels = np.concatenate([corner.reshape(-1, 3) for corner in corners], axis=0)
    corner_pixels = corner_pixels.reshape(-1, 1, 3).astype(np.uint8)
    lab_pixels = cv2.cvtColor(corner_pixels, cv2.COLOR_RGB2LAB).reshape(-1, 3)
    return np.median(lab_pixels, axis=0)


def segment_lesion_lab(
    image: Image.Image,
    min_area_ratio: float = 0.002,
) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[Image.Image], Dict[str, float]]:
    rgb = pil_to_rgb_array(image)
    h, w, _ = rgb.shape
    diagnostics = {"area_ratio": 0.0, "touches_border": 0.0}

    if h < 20 or w < 20:
        return None, None, diagnostics

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    bg_lab = estimate_background_lab(rgb).astype(np.float32)
    dist = np.linalg.norm(lab - bg_lab.reshape(1, 1, 3), axis=2)
    dist = cv2.GaussianBlur(dist, (0, 0), sigmaX=2.0)
    dist_norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    _, mask = cv2.threshold(dist_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return None, Image.fromarray(mask), diagnostics

    image_area = h * w
    min_area = image_area * min_area_ratio
    components = []
    for label in range(1, num_labels):
        x, y, bw, bh, area = stats[label]
        if area >= min_area:
            components.append((area, x, y, bw, bh))

    if not components:
        return None, Image.fromarray(mask), diagnostics

    area, x, y, bw, bh = max(components, key=lambda item: item[0])
    diagnostics["area_ratio"] = float(area / image_area)
    diagnostics["touches_border"] = float(x <= 2 or y <= 2 or x + bw >= w - 2 or y + bh >= h - 2)

    box = (int(x), int(y), int(x + bw), int(y + bh))
    box = expand_box(box, image_size=(w, h), padding_ratio=0.18)
    return box, Image.fromarray(mask), diagnostics


def build_manual_box_controls(
    image: Image.Image,
    initial_box: Optional[Tuple[int, int, int, int]],
) -> Tuple[Image.Image, Optional[Tuple[int, int, int, int]]]:
    width, height = image.size
    if initial_box is None:
        initial_box = (int(width * 0.20), int(height * 0.20), int(width * 0.80), int(height * 0.80))

    x_min_default, y_min_default, x_max_default, y_max_default = initial_box
    st.caption("Inclua toda a lesão e um pouco de pele ao redor. Evite cortar bordas importantes da lesão.")

    col_x, col_y = st.columns(2)
    with col_x:
        x_min = st.slider("x mínimo", 0, max(0, width - 1), int(x_min_default))
        x_max = st.slider("x máximo", 1, width, int(x_max_default))
    with col_y:
        y_min = st.slider("y mínimo", 0, max(0, height - 1), int(y_min_default))
        y_max = st.slider("y máximo", 1, height, int(y_max_default))

    if x_max <= x_min or y_max <= y_min:
        st.error("Recorte inválido. Ajuste os valores para que x máximo/y máximo sejam maiores.")
        return image, None

    box = (x_min, y_min, x_max, y_max)
    return crop_image_by_box(image, box), box


# ============================================================
# Interpretability
# ============================================================

def find_last_conv_layer(model: keras.Model) -> str:
    for layer in reversed(model.layers):
        try:
            if len(layer.output.shape) == 4:
                return layer.name
        except Exception:
            continue
    raise ValueError("Não foi possível encontrar uma camada convolucional 4D acessível.")


def make_gradcam_heatmap(
    input_array: np.ndarray,
    model: keras.Model,
    last_conv_layer_name: str,
    pred_index: Optional[int] = None,
) -> np.ndarray:
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output],
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(input_array, training=False)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    if grads is None:
        raise ValueError("Gradientes nulos para Grad-CAM.")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    max_value = tf.reduce_max(heatmap)
    if max_value == 0:
        raise ValueError("Heatmap Grad-CAM vazio.")
    return (heatmap / max_value).numpy()


def make_saliency_heatmap(
    input_array: np.ndarray,
    model: keras.Model,
    pred_index: Optional[int] = None,
) -> np.ndarray:
    input_tensor = tf.convert_to_tensor(input_array)
    with tf.GradientTape() as tape:
        tape.watch(input_tensor)
        predictions = call_model_for_gradients(model, input_tensor)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_score = predictions[:, pred_index]
    grads = tape.gradient(class_score, input_tensor)
    if grads is None:
        raise ValueError("Não foi possível calcular saliency map.")
    saliency = tf.reduce_max(tf.abs(grads), axis=-1)[0]
    saliency = saliency - tf.reduce_min(saliency)
    saliency = saliency / (tf.reduce_max(saliency) + 1e-8)
    return saliency.numpy()


def overlay_heatmap_on_image(
    original_image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.40,
) -> Image.Image:
    image = original_image.convert("RGB")
    image_np = np.array(image)
    heatmap_resized = cv2.resize(heatmap, (image_np.shape[1], image_np.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(image_np, 1 - alpha, heatmap_color, alpha, 0)
    return Image.fromarray(overlay)


def extract_bounding_box_from_heatmap(
    heatmap: np.ndarray,
    original_size: Tuple[int, int],
    threshold: float,
    min_area_ratio: float = 0.003,
    ignore_border_ratio: float = 0.03,
) -> Optional[Tuple[int, int, int, int]]:
    width, height = original_size
    heatmap_resized = cv2.resize(heatmap, (width, height)).astype(np.float32)
    heatmap_resized = cv2.GaussianBlur(heatmap_resized, (0, 0), sigmaX=8.0)
    max_value = float(np.max(heatmap_resized))
    if max_value <= 1e-8:
        return None
    heatmap_resized = heatmap_resized / max_value

    border_x = int(width * ignore_border_ratio)
    border_y = int(height * ignore_border_ratio)
    if border_y > 0:
        heatmap_resized[:border_y, :] = 0
        heatmap_resized[-border_y:, :] = 0
    if border_x > 0:
        heatmap_resized[:, :border_x] = 0
        heatmap_resized[:, -border_x:] = 0

    mask = (heatmap_resized >= threshold).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return None

    min_area = width * height * min_area_ratio
    components = []
    for label in range(1, num_labels):
        x, y, bw, bh, area = stats[label]
        if area >= min_area:
            components.append((area, x, y, bw, bh))
    if not components:
        return None
    _, x, y, bw, bh = max(components, key=lambda item: item[0])
    box = (int(x), int(y), int(x + bw), int(y + bh))
    return expand_box(box, image_size=(width, height), padding_ratio=0.15)


def render_interpretability_section(
    model: keras.Model,
    image_for_model: Image.Image,
    final_class: str,
    bbox_threshold: float,
) -> None:
    st.markdown("---")
    st.markdown("## Interpretabilidade visual")
    st.info(
        "A visualização mostra regiões que influenciaram a decisão do modelo. Quando Grad-CAM não está disponível, "
        "o app usa saliency map. A bounding box é aproximada e não equivale a segmentação clínica real."
    )

    input_array = preprocess_image(image_for_model)
    final_class_index = CLASS_NAMES.index(final_class)

    try:
        last_conv_layer_name = find_last_conv_layer(model)
        heatmap_final = make_gradcam_heatmap(input_array, model, last_conv_layer_name, final_class_index)
        method_final = "Grad-CAM"
    except Exception:
        heatmap_final = make_saliency_heatmap(input_array, model, final_class_index)
        method_final = "Saliency map"

    heatmap_mel = make_saliency_heatmap(input_array, model, MEL_INDEX)

    overlay_final = overlay_heatmap_on_image(image_for_model, heatmap_final, alpha=0.40)
    bbox_final = extract_bounding_box_from_heatmap(heatmap_final, image_for_model.size, bbox_threshold)
    boxed_final = draw_box_on_image(image_for_model, bbox_final, f"Região: {final_class}")

    overlay_mel = overlay_heatmap_on_image(image_for_model, heatmap_mel, alpha=0.40)
    bbox_mel = extract_bounding_box_from_heatmap(heatmap_mel, image_for_model.size, bbox_threshold)
    boxed_mel = draw_box_on_image(image_for_model, bbox_mel, "Região: melanoma")

    col_a, col_b = st.columns(2)
    with col_a:
        st.image(overlay_final, caption=f"{method_final} da classe prevista ({final_class})", width="stretch")
    with col_b:
        st.image(boxed_final, caption="Região aproximada mais influente para a decisão", width="stretch")

    col_c, col_d = st.columns(2)
    with col_c:
        st.image(overlay_mel, caption="Mapa de atenção específico para melanoma", width="stretch")
    with col_d:
        st.image(boxed_mel, caption="Região aproximada associada à probabilidade de melanoma", width="stretch")


# ============================================================
# Rendering: model info, results, probabilities, report
# ============================================================

def render_header() -> None:
    st.markdown('<div class="main-title">Skin Lesion Triage Support — v3</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Protótipo web de apoio à triagem inicial de lesões de pele, com foco em sensibilidade para melanoma.</div>',
        unsafe_allow_html=True,
    )
    st.warning(
        "Este sistema é uma prova de conceito acadêmica. Ele não realiza diagnóstico médico, "
        "não substitui dermatologistas e não deve ser usado como ferramenta clínica definitiva."
    )


def render_model_summary(metrics: Dict, policy: Dict) -> None:
    with st.expander("Sobre o modelo v3", expanded=False):
        st.write(
            "O modelo v3 usa EfficientNetB0, focal loss, data augmentation, class weights, melanoma boost e uma "
            "política melanoma-sensitive para priorizar recall de melanoma."
        )

        rows = [
            ("Accuracy", metrics.get("accuracy")),
            ("Recall melanoma", metrics.get("recall_melanoma")),
            ("Precision melanoma", metrics.get("precision_melanoma")),
            ("F1 melanoma", metrics.get("f1_melanoma")),
            ("F1 macro", metrics.get("f1_macro")),
            ("ROC-AUC OVR macro", metrics.get("roc_auc_ovr_macro")),
            ("Threshold melanoma", policy.get("melanoma_threshold")),
        ]
        df = pd.DataFrame(
            [{"Métrica": name, "Valor": "—" if value is None else f"{float(value):.4f}"} for name, value in rows]
        )
        st.dataframe(df, width="stretch", hide_index=True)
        st.caption(
            "O threshold principal foi definido na validação do modelo v3 para priorizar a sensibilidade de melanoma."
        )


def render_result_cards(
    final_class: str,
    final_prob: float,
    melanoma_prob: float,
    risk_level: str,
) -> None:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Classe final</div>
                <div class="metric-value">{final_class} — {CLASS_DESCRIPTIONS_PT[final_class]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Probabilidade de melanoma</div>
                <div class="metric-value">{format_percentage(melanoma_prob)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Nível de triagem</div>
                <div class="metric-value">{risk_label(risk_level)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_risk_panel(
    risk_level: str,
    melanoma_prob: float,
    melanoma_threshold: float,
    intermediate_threshold: float,
    final_class: str,
) -> None:
    if risk_level == "high":
        css = "risk-high"
        text = (
            f"A probabilidade de melanoma foi {format_percentage(melanoma_prob)}, acima do threshold principal "
            f"de {format_percentage(melanoma_threshold)}. A imagem foi sinalizada como suspeita para melanoma."
        )
    elif risk_level == "intermediate":
        css = "risk-mid"
        text = (
            f"A probabilidade de melanoma foi {format_percentage(melanoma_prob)}, abaixo do threshold principal "
            f"de {format_percentage(melanoma_threshold)}, mas acima do limiar intermediário de "
            f"{format_percentage(intermediate_threshold)}. Recomenda-se cautela em triagem."
        )
    else:
        css = "risk-low"
        text = (
            f"A probabilidade de melanoma foi {format_percentage(melanoma_prob)}, abaixo do limiar intermediário "
            f"de {format_percentage(intermediate_threshold)} e do threshold principal de {format_percentage(melanoma_threshold)}."
        )

    st.markdown(f'<div class="{css}">{text}</div>', unsafe_allow_html=True)
    st.write(triage_message(risk_level, final_class))


def render_class_probabilities(probs_by_class: Dict[str, float]) -> None:
    sorted_probs = sorted(probs_by_class.items(), key=lambda item: item[1], reverse=True)
    st.markdown("---")
    st.markdown("## Top 3 predições")
    for rank, (class_name, prob) in enumerate(sorted_probs[:3], start=1):
        st.write(f"{rank}. **{class_name} — {CLASS_DESCRIPTIONS_PT[class_name]}**: {format_percentage(prob)}")

    with st.expander("Ver todas as probabilidades por classe", expanded=False):
        for class_name, prob in sorted_probs:
            label = f"{class_name} — {CLASS_DESCRIPTIONS_PT[class_name]} ({CLASS_DESCRIPTIONS[class_name]})"
            st.progress(float(prob), text=f"{label}: {format_percentage(prob)}")


def build_prediction_report(
    file_name: str,
    selected_mode: str,
    final_class: str,
    final_prob: float,
    melanoma_prob: float,
    risk_level: str,
    argmax_class: str,
    argmax_prob: float,
    melanoma_threshold: float,
    intermediate_threshold: float,
    probs_by_class: Dict[str, float],
) -> str:
    lines = [
        "Skin Lesion Triage Support — v3",
        "Resultado da triagem",
        "",
        f"Data/hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Arquivo: {file_name}",
        f"Entrada utilizada: {selected_mode}",
        "",
        f"Classe final: {final_class} — {CLASS_DESCRIPTIONS_PT[final_class]}",
        f"Probabilidade da classe final: {format_percentage(final_prob)}",
        f"Probabilidade de melanoma: {format_percentage(melanoma_prob)}",
        f"Nível de triagem: {risk_label(risk_level)}",
        f"Classe argmax antes da regra: {argmax_class} — {format_percentage(argmax_prob)}",
        "",
        f"Threshold melanoma-sensitive: {format_percentage(melanoma_threshold)}",
        f"Threshold risco intermediário: {format_percentage(intermediate_threshold)}",
        "",
        "Probabilidades por classe:",
    ]
    for cls, prob in sorted(probs_by_class.items(), key=lambda item: item[1], reverse=True):
        lines.append(f"- {cls} — {CLASS_DESCRIPTIONS_PT[cls]}: {format_percentage(prob)}")

    lines.extend([
        "",
        "Aviso metodológico:",
        "Este resultado é apenas apoio à triagem inicial. Não é diagnóstico médico e não substitui avaliação profissional.",
    ])
    return "\n".join(lines)


def predict_single_mode(
    model: keras.Model,
    image: Image.Image,
    mode_label: str,
    melanoma_threshold: float,
    intermediate_threshold: float,
) -> Dict:
    probs = predict_probabilities(model, image)
    final_class, final_prob, argmax_class, argmax_prob = apply_melanoma_policy(probs, melanoma_threshold)
    melanoma_prob = probs["mel"]
    risk_level = get_melanoma_risk_level(melanoma_prob, melanoma_threshold, intermediate_threshold)
    return {
        "mode": mode_label,
        "image": image,
        "probs": probs,
        "final_class": final_class,
        "final_prob": final_prob,
        "argmax_class": argmax_class,
        "argmax_prob": argmax_prob,
        "melanoma_prob": melanoma_prob,
        "risk_level": risk_level,
    }


def comparison_table(results: List[Dict]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "Entrada": r["mode"],
            "Classe final": f"{r['final_class']} — {CLASS_DESCRIPTIONS_PT[r['final_class']]}",
            "Prob. classe final": format_percentage(r["final_prob"]),
            "Prob. melanoma": format_percentage(r["melanoma_prob"]),
            "Nível": risk_label(r["risk_level"]),
        }
        for r in results
    ])


# ============================================================
# Sidebar
# ============================================================

render_header()

with st.sidebar:
    st.header("Configurações")
    st.markdown("### Caminhos")
    model_path = st.text_input("Caminho do modelo v3", value=str(DEFAULT_MODEL_PATH))
    policy_path_text = st.text_input("Caminho da política v3", value=str(DEFAULT_POLICY_PATH))
    metrics_path_text = st.text_input("Caminho das métricas v3", value=str(DEFAULT_METRICS_PATH))

    policy = load_decision_policy(Path(policy_path_text))
    metrics = load_json_file(metrics_path_text)

    loaded_threshold = float(policy.get("melanoma_threshold", DEFAULT_MELANOMA_THRESHOLD))

    with st.expander("Configurações avançadas", expanded=False):
        melanoma_threshold = st.slider(
            "Threshold melanoma-sensitive",
            min_value=0.05,
            max_value=0.90,
            value=loaded_threshold,
            step=0.01,
        )
        intermediate_threshold = st.slider(
            "Threshold de risco intermediário",
            min_value=0.01,
            max_value=max(0.02, float(melanoma_threshold) - 0.01),
            value=min(DEFAULT_INTERMEDIATE_THRESHOLD, float(melanoma_threshold) - 0.01),
            step=0.01,
        )
        bbox_threshold = st.slider(
            "Threshold da bounding box aproximada",
            min_value=0.30,
            max_value=0.90,
            value=DEFAULT_BBOX_THRESHOLD,
            step=0.05,
        )
        show_interpretability = st.checkbox("Mostrar interpretabilidade visual", value=True)
        compare_modes = st.checkbox("Comparar modos de entrada", value=False)
    if "bbox_threshold" not in locals():
        bbox_threshold = DEFAULT_BBOX_THRESHOLD
        show_interpretability = True
        compare_modes = False

    st.markdown("---")
    st.markdown("### Classes do modelo")
    for class_name in CLASS_NAMES:
        st.write(f"**{class_name}** — {CLASS_DESCRIPTIONS_PT[class_name]}")

render_model_summary(metrics, policy)

# ============================================================
# Main app
# ============================================================

st.markdown("## 1. Envie a imagem")
uploaded_file = st.file_uploader("Envie uma imagem de lesão de pele", type=["jpg", "jpeg", "png"])

if uploaded_file is None:
    st.info("Envie uma imagem para começar a triagem.")
    st.stop()

original_image = Image.open(uploaded_file).convert("RGB")
render_quality_section(original_image)

st.markdown("## 2. Escolha a entrada para o modelo")
st.markdown(
    '<div class="method-note">Modo recomendado: <b>recorte manual</b>. Use o recorte automático como apoio experimental e a imagem inteira apenas para comparação.</div>',
    unsafe_allow_html=True,
)

auto_box, auto_mask, auto_diag = segment_lesion_lab(original_image)

mode = st.radio(
    "Entrada para classificação",
    options=["Recorte manual", "Recorte automático aproximado", "Imagem inteira"],
    horizontal=True,
)

selected_image = original_image
selected_box: Optional[Tuple[int, int, int, int]] = None
selected_mode = mode
manual_image: Optional[Image.Image] = None
manual_box: Optional[Tuple[int, int, int, int]] = None

auto_image = crop_image_by_box(original_image, auto_box) if auto_box is not None else original_image

if mode == "Recorte manual":
    manual_image, manual_box = build_manual_box_controls(original_image, auto_box)
    selected_image = manual_image
    selected_box = manual_box
    selected_mode = "Recorte manual"
    c1, c2 = st.columns(2)
    with c1:
        st.image(draw_box_on_image(original_image, manual_box, "Recorte manual"), caption="Região selecionada", width="stretch")
    with c2:
        st.image(selected_image, caption="Imagem enviada ao modelo", width="stretch")

elif mode == "Recorte automático aproximado":
    if auto_box is None:
        st.warning("Não foi possível gerar recorte automático. A imagem inteira será usada.")
        selected_image = original_image
        selected_mode = "Imagem inteira"
    else:
        selected_image = auto_image
        selected_box = auto_box
        c1, c2 = st.columns(2)
        with c1:
            st.image(draw_box_on_image(original_image, auto_box, "Recorte automático"), caption="Bounding box aproximada por cor", width="stretch")
        with c2:
            st.image(selected_image, caption="Imagem enviada ao modelo", width="stretch")
        if auto_diag.get("touches_border"):
            st.warning("O recorte automático toca a borda da imagem. Verifique se a lesão não foi cortada.")
        if auto_diag.get("area_ratio", 0) > 0.65:
            st.warning("O recorte automático ficou muito grande. Pode ter incluído fundo em excesso.")
        with st.expander("Ver máscara automática aproximada", expanded=False):
            if auto_mask is not None:
                st.image(auto_mask, caption="Máscara LAB + Otsu", width="stretch")
else:
    selected_image = original_image
    selected_mode = "Imagem inteira"
    st.image(original_image, caption="Imagem inteira enviada ao modelo", width="stretch")

st.markdown("## 3. Execute a predição")
run_prediction = st.button("Executar predição", type="primary")

if not run_prediction:
    st.info("Clique em **Executar predição** para gerar o resultado.")
    st.stop()

try:
    with st.spinner("Carregando modelo v3 e executando predição..."):
        model = load_trained_model(model_path)
except Exception as exc:
    st.error("Não foi possível carregar o modelo v3. Verifique o caminho do arquivo .keras.")
    with st.expander("Detalhes técnicos do erro"):
        st.exception(exc)
    st.stop()

results_to_compare: List[Dict] = []

try:
    selected_result = predict_single_mode(
        model=model,
        image=selected_image,
        mode_label=selected_mode,
        melanoma_threshold=melanoma_threshold,
        intermediate_threshold=intermediate_threshold,
    )

    if compare_modes:
        results_to_compare.append(
            predict_single_mode(model, original_image, "Imagem inteira", melanoma_threshold, intermediate_threshold)
        )
        if auto_box is not None:
            results_to_compare.append(
                predict_single_mode(model, auto_image, "Recorte automático", melanoma_threshold, intermediate_threshold)
            )
        if manual_image is not None:
            results_to_compare.append(
                predict_single_mode(model, manual_image, "Recorte manual", melanoma_threshold, intermediate_threshold)
            )

except Exception as exc:
    st.error("Não foi possível executar a predição.")
    with st.expander("Detalhes técnicos do erro"):
        st.exception(exc)
    st.stop()

st.markdown("---")
st.markdown("## Resultado da triagem")

col_image, col_result = st.columns([1, 1.15])
with col_image:
    st.image(selected_image, caption=f"Imagem usada pelo modelo: {selected_mode}", width="stretch")

with col_result:
    render_result_cards(
        final_class=selected_result["final_class"],
        final_prob=selected_result["final_prob"],
        melanoma_prob=selected_result["melanoma_prob"],
        risk_level=selected_result["risk_level"],
    )
    st.markdown("### Regra melanoma-sensitive")
    render_risk_panel(
        risk_level=selected_result["risk_level"],
        melanoma_prob=selected_result["melanoma_prob"],
        melanoma_threshold=melanoma_threshold,
        intermediate_threshold=intermediate_threshold,
        final_class=selected_result["final_class"],
    )
    st.markdown("### Comparação com argmax normal")
    st.write(
        f"A classe por maior probabilidade antes da regra melanoma-sensitive era: "
        f"**{selected_result['argmax_class']} — {CLASS_DESCRIPTIONS_PT[selected_result['argmax_class']]}** "
        f"({format_percentage(selected_result['argmax_prob'])})."
    )
    st.markdown("### Entrada utilizada")
    if selected_mode == "Imagem inteira":
        st.write("A predição foi feita na imagem inteira. Isso pode ser mais sensível a fundo, iluminação e regiões fora da lesão.")
    else:
        st.write(f"A predição foi feita usando **{selected_mode}** para reduzir a influência de fundo e regiões não relevantes.")

if compare_modes and results_to_compare:
    st.markdown("---")
    st.markdown("## Comparação entre modos de entrada")
    st.dataframe(comparison_table(results_to_compare), width="stretch", hide_index=True)
    st.caption("Diferenças entre modos mostram como o recorte influencia a predição. Para apresentação, prefira o recorte manual como modo recomendado.")

render_class_probabilities(selected_result["probs"])

report_text = build_prediction_report(
    file_name=uploaded_file.name,
    selected_mode=selected_mode,
    final_class=selected_result["final_class"],
    final_prob=selected_result["final_prob"],
    melanoma_prob=selected_result["melanoma_prob"],
    risk_level=selected_result["risk_level"],
    argmax_class=selected_result["argmax_class"],
    argmax_prob=selected_result["argmax_prob"],
    melanoma_threshold=melanoma_threshold,
    intermediate_threshold=intermediate_threshold,
    probs_by_class=selected_result["probs"],
)

st.download_button(
    "Baixar resultado da triagem (.txt)",
    data=report_text,
    file_name=f"resultado_triagem_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
    mime="text/plain",
)

if show_interpretability:
    try:
        render_interpretability_section(
            model=model,
            image_for_model=selected_image,
            final_class=selected_result["final_class"],
            bbox_threshold=bbox_threshold,
        )
    except Exception as exc:
        st.warning("Não foi possível gerar interpretabilidade visual para esta imagem.")
        with st.expander("Detalhes técnicos do erro"):
            st.exception(exc)

st.markdown("---")
st.markdown(
    """
## Observação metodológica

O modelo foi treinado com o dataset **HAM10000**, composto majoritariamente por imagens dermatoscópicas. Portanto, apesar da proposta futura envolver imagens de smartphone, esta versão deve ser interpretada como **prova de conceito acadêmica**.

Esta versão do app adiciona melhorias para reduzir limitações observadas nos testes:

1. **Modelo v3 melanoma-sensitive** com threshold principal otimizado para aumentar sensibilidade.
2. **Faixa intermediária de risco** para evitar tratar casos suspeitos abaixo do threshold como simplesmente negativos.
3. **Recorte manual recomendado** e recorte automático aproximado para reduzir influência do fundo.
4. **Comparação entre modos de entrada**, mostrando como o recorte pode alterar o resultado.
5. **Interpretabilidade visual** com Grad-CAM ou saliency map, sem afirmar segmentação clínica real.

A saída deve ser usada apenas como apoio à triagem inicial e não como diagnóstico definitivo.
"""
)
