
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageDraw
from tensorflow import keras


# ============================================================
# Configurações principais
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "reports"
    / "model_v2_mel_sensitive_final"
    / "cnn_ham10000_v2.keras"
)

DEFAULT_POLICY_PATH = (
    PROJECT_ROOT
    / "reports"
    / "model_v2_mel_sensitive_final"
    / "decision_policy_v2.json"
)

IMAGE_SIZE = (224, 224)

CLASS_NAMES: List[str] = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc",
]

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

DEFAULT_MELANOMA_THRESHOLD = 0.36
DEFAULT_INTERMEDIATE_THRESHOLD = 0.15
DEFAULT_BBOX_THRESHOLD = 0.60
MEL_INDEX = CLASS_NAMES.index("mel")


# ============================================================
# Funções auxiliares gerais
# ============================================================

@st.cache_resource
def load_trained_model(model_path: str) -> keras.Model:
    path = Path(model_path)

    if not path.exists():
        raise FileNotFoundError(f"Modelo não encontrado em: {path}")

    return keras.models.load_model(path)


def load_decision_policy(policy_path: Path) -> Dict[str, float]:
    if not policy_path.exists():
        return {"melanoma_threshold": DEFAULT_MELANOMA_THRESHOLD}

    try:
        with policy_path.open("r", encoding="utf-8") as file:
            policy = json.load(file)

        if "melanoma_threshold" not in policy:
            policy["melanoma_threshold"] = DEFAULT_MELANOMA_THRESHOLD

        return policy

    except Exception:
        return {"melanoma_threshold": DEFAULT_MELANOMA_THRESHOLD}


def format_percentage(value: float) -> str:
    return f"{value * 100:.2f}%"


def pil_to_rgb_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"))


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Pré-processamento para o modelo.

    O modelo v2 foi treinado recebendo pixels 0-255 e o preprocess_input da
    EfficientNet foi incluído dentro da arquitetura treinada. Por isso, aqui
    mantemos a imagem em float32 na escala 0-255.
    """
    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE)

    array = np.asarray(image).astype(np.float32)
    array = np.expand_dims(array, axis=0)

    return array


def predict_image(
    model: keras.Model,
    image: Image.Image,
    melanoma_threshold: float,
) -> Tuple[str, float, Dict[str, float], str, float]:
    input_array = preprocess_image(image)
    probabilities = model.predict(input_array, verbose=0)[0]

    probs_by_class = {
        class_name: float(prob)
        for class_name, prob in zip(CLASS_NAMES, probabilities)
    }

    argmax_idx = int(np.argmax(probabilities))
    argmax_class = CLASS_NAMES[argmax_idx]
    melanoma_prob = probs_by_class["mel"]

    if melanoma_prob >= melanoma_threshold:
        final_class = "mel"
        final_prob = melanoma_prob
    else:
        final_class = argmax_class
        final_prob = probs_by_class[argmax_class]

    return final_class, final_prob, probs_by_class, argmax_class, melanoma_prob


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


def get_triage_message(
    final_class: str,
    melanoma_prob: float,
    melanoma_threshold: float,
    intermediate_threshold: float,
) -> str:
    risk_level = get_melanoma_risk_level(
        melanoma_prob=melanoma_prob,
        melanoma_threshold=melanoma_threshold,
        intermediate_threshold=intermediate_threshold,
    )

    if risk_level == "high":
        return (
            "A imagem foi marcada como suspeita para melanoma pela regra "
            "melanoma-sensitive. Recomenda-se avaliação médica especializada."
        )

    if risk_level == "intermediate":
        return (
            "A probabilidade de melanoma ficou em uma faixa intermediária. "
            "Embora a classe final não tenha sido ajustada para melanoma, este resultado "
            "deve ser tratado com cautela em triagem, principalmente se a lesão apresentar "
            "assimetria, bordas irregulares, variação de cor, crescimento ou sintomas."
        )

    return (
        "A imagem não foi marcada como melanoma pela regra de triagem. "
        "Mesmo assim, o resultado não substitui avaliação médica."
    )


def render_class_probabilities(probs_by_class: Dict[str, float]) -> None:
    st.markdown("---")
    st.markdown("## Probabilidades por classe")

    sorted_probs = sorted(
        probs_by_class.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    for class_name, prob in sorted_probs:
        st.progress(float(prob))
        st.write(
            f"**{class_name}** — {CLASS_DESCRIPTIONS_PT[class_name]} "
            f"({CLASS_DESCRIPTIONS[class_name]}): {format_percentage(prob)}"
        )

    st.markdown("---")
    st.markdown("## Top 3 predições")

    for rank, (class_name, prob) in enumerate(sorted_probs[:3], start=1):
        st.write(
            f"{rank}. **{class_name}** — {CLASS_DESCRIPTIONS_PT[class_name]} "
            f"({format_percentage(prob)})"
        )


# ============================================================
# Qualidade de imagem e preparação da entrada
# ============================================================

def compute_image_quality(image: Image.Image) -> Dict[str, float]:
    """
    Mede alguns indicadores simples de qualidade.
    Não bloqueia a predição; apenas gera avisos.
    """
    rgb = pil_to_rgb_array(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    return {
        "brightness": brightness,
        "contrast": contrast,
        "blur_score": blur_score,
    }


def render_quality_warnings(image: Image.Image) -> None:
    quality = compute_image_quality(image)

    warnings = []

    if quality["brightness"] < 45:
        warnings.append("A imagem parece escura.")
    elif quality["brightness"] > 220:
        warnings.append("A imagem parece muito clara ou superexposta.")

    if quality["contrast"] < 20:
        warnings.append("A imagem parece ter baixo contraste.")

    if quality["blur_score"] < 25:
        warnings.append("A imagem pode estar desfocada.")

    if warnings:
        st.warning(
            "Possíveis problemas de qualidade da imagem: "
            + " ".join(warnings)
            + " Isso pode reduzir a confiabilidade da predição."
        )
    else:
        st.success("Indicadores básicos de qualidade da imagem estão aceitáveis.")


def crop_image_by_box(
    image: Image.Image,
    box: Tuple[int, int, int, int],
) -> Image.Image:
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

    bw = x_max - x_min
    bh = y_max - y_min

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


# ============================================================
# Segmentação clássica aproximada por cor
# ============================================================

def estimate_background_lab(rgb: np.ndarray, corner_fraction: float = 0.10) -> np.ndarray:
    """
    Estima a cor de fundo usando os cantos da imagem.
    A ideia segue a lógica de usar a pele ao redor como referência de fundo.
    """
    h, w, _ = rgb.shape
    ch = max(5, int(h * corner_fraction))
    cw = max(5, int(w * corner_fraction))

    corners = [
        rgb[:ch, :cw],
        rgb[:ch, -cw:],
        rgb[-ch:, :cw],
        rgb[-ch:, -cw:],
    ]

    corner_pixels = np.concatenate([c.reshape(-1, 3) for c in corners], axis=0)
    corner_pixels = corner_pixels.reshape(-1, 1, 3).astype(np.uint8)
    lab_pixels = cv2.cvtColor(corner_pixels, cv2.COLOR_RGB2LAB).reshape(-1, 3)

    return np.median(lab_pixels, axis=0)


def segment_lesion_lab(
    image: Image.Image,
    min_area_ratio: float = 0.002,
) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[Image.Image]]:
    """
    Segmentação clássica aproximada:
    1. converte RGB para LAB;
    2. estima o fundo pelos cantos;
    3. calcula distância de cor em relação ao fundo;
    4. aplica Otsu;
    5. limpa máscara;
    6. pega maior componente conectado.

    Não é segmentação clínica validada, mas ajuda a gerar um recorte melhor.
    """
    rgb = pil_to_rgb_array(image)
    h, w, _ = rgb.shape

    if h < 20 or w < 20:
        return None, None

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    bg_lab = estimate_background_lab(rgb).astype(np.float32)

    dist = np.linalg.norm(lab - bg_lab.reshape(1, 1, 3), axis=2)
    dist = cv2.GaussianBlur(dist, (0, 0), sigmaX=2.0)

    dist_norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    _, mask = cv2.threshold(
        dist_norm,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if num_labels <= 1:
        return None, Image.fromarray(mask)

    image_area = h * w
    min_area = image_area * min_area_ratio

    components = []
    for label in range(1, num_labels):
        x, y, bw, bh, area = stats[label]
        if area >= min_area:
            components.append((area, x, y, bw, bh))

    if not components:
        return None, Image.fromarray(mask)

    area, x, y, bw, bh = max(components, key=lambda item: item[0])

    box = (int(x), int(y), int(x + bw), int(y + bh))
    box = expand_box(box, image_size=(w, h), padding_ratio=0.18)

    return box, Image.fromarray(mask)


def render_crop_controls(image: Image.Image) -> Tuple[Image.Image, Optional[Tuple[int, int, int, int]], str]:
    """
    Permite escolher qual imagem será enviada ao modelo:
    - imagem inteira;
    - recorte automático por segmentação LAB;
    - recorte manual com sliders.
    """
    st.markdown("## Preparação da imagem")

    st.info(
        "Para reduzir a influência de fundo, iluminação e regiões que não pertencem à lesão, "
        "o app permite classificar a imagem inteira, um recorte automático aproximado ou um recorte manual."
    )

    auto_box, auto_mask = segment_lesion_lab(image)

    mode = st.radio(
        "Escolha a entrada para a classificação",
        options=[
            "Imagem inteira",
            "Recorte automático aproximado",
            "Recorte manual",
        ],
        horizontal=True,
    )

    selected_box: Optional[Tuple[int, int, int, int]] = None
    selected_image = image
    selected_mode = mode

    if mode == "Recorte automático aproximado":
        if auto_box is None:
            st.warning(
                "Não foi possível gerar um recorte automático confiável. "
                "A imagem inteira será usada."
            )
            selected_image = image
            selected_box = None
            selected_mode = "Imagem inteira"
        else:
            selected_box = auto_box
            selected_image = crop_image_by_box(image, selected_box)

            col_a, col_b = st.columns(2)
            with col_a:
                st.image(
                    draw_box_on_image(image, selected_box, "Recorte automático"),
                    caption="Bounding box por segmentação clássica aproximada",
                    use_container_width=True,
                )
            with col_b:
                st.image(
                    selected_image,
                    caption="Recorte enviado ao modelo",
                    use_container_width=True,
                )

            if auto_mask is not None:
                with st.expander("Ver máscara aproximada por cor"):
                    st.image(auto_mask, caption="Máscara aproximada LAB + Otsu", use_container_width=True)

    elif mode == "Recorte manual":
        st.write("Ajuste os sliders para selecionar a região da lesão.")

        width, height = image.size

        default_box = auto_box
        if default_box is None:
            default_box = (
                int(width * 0.20),
                int(height * 0.20),
                int(width * 0.80),
                int(height * 0.80),
            )

        x_min_default, y_min_default, x_max_default, y_max_default = default_box

        x_min = st.slider("x mínimo", 0, max(0, width - 1), int(x_min_default))
        x_max = st.slider("x máximo", 1, width, int(x_max_default))
        y_min = st.slider("y mínimo", 0, max(0, height - 1), int(y_min_default))
        y_max = st.slider("y máximo", 1, height, int(y_max_default))

        if x_max <= x_min or y_max <= y_min:
            st.error("Recorte inválido. Ajuste os valores para que x máximo/y máximo sejam maiores.")
            selected_image = image
            selected_box = None
        else:
            selected_box = (x_min, y_min, x_max, y_max)
            selected_image = crop_image_by_box(image, selected_box)

            col_a, col_b = st.columns(2)
            with col_a:
                st.image(
                    draw_box_on_image(image, selected_box, "Recorte manual"),
                    caption="Região selecionada manualmente",
                    use_container_width=True,
                )
            with col_b:
                st.image(
                    selected_image,
                    caption="Recorte enviado ao modelo",
                    use_container_width=True,
                )

    else:
        st.image(image, caption="Imagem inteira enviada ao modelo", use_container_width=True)

    return selected_image, selected_box, selected_mode


# ============================================================
# Grad-CAM, saliency map e bounding box melhorada
# ============================================================

def find_last_conv_layer(model: keras.Model) -> str:
    """
    Tenta encontrar automaticamente uma camada convolucional 4D no modelo principal.
    Em alguns modelos salvos com EfficientNet encapsulada e pooling='avg', o Grad-CAM
    real pode não estar disponível; nesse caso, o app usa saliency map como fallback.
    """
    for layer in reversed(model.layers):
        try:
            output_shape = layer.output.shape
            if len(output_shape) == 4:
                return layer.name
        except Exception:
            continue

    raise ValueError(
        "Não foi possível encontrar automaticamente uma camada convolucional 4D."
    )


def make_gradcam_heatmap(
    input_array: np.ndarray,
    model: keras.Model,
    last_conv_layer_name: str,
    pred_index: Optional[int] = None,
) -> np.ndarray:
    """
    Tenta gerar Grad-CAM. Se a arquitetura salva não permitir gradientes nessa
    camada, uma exceção será levantada e o app usará saliency map.
    """
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output,
        ],
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

    heatmap = heatmap / max_value
    return heatmap.numpy()


def make_saliency_heatmap(
    input_array: np.ndarray,
    model: keras.Model,
    pred_index: Optional[int] = None,
) -> np.ndarray:
    """
    Fallback: mapa de saliência baseado no gradiente da classe em relação à imagem.
    Funciona mesmo quando não é possível acessar a última camada convolucional.
    """
    input_tensor = tf.convert_to_tensor(input_array)

    with tf.GradientTape() as tape:
        tape.watch(input_tensor)
        predictions = model(input_tensor, training=False)

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

    heatmap_resized = cv2.resize(
        heatmap,
        (image_np.shape[1], image_np.shape[0]),
    )

    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(
        image_np,
        1 - alpha,
        heatmap_color,
        alpha,
        0,
    )

    return Image.fromarray(overlay)


def extract_bounding_box_from_heatmap(
    heatmap: np.ndarray,
    original_size: Tuple[int, int],
    threshold: float = DEFAULT_BBOX_THRESHOLD,
    min_area_ratio: float = 0.003,
    ignore_border_ratio: float = 0.03,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Bounding box mais estável:
    - redimensiona heatmap;
    - suaviza;
    - ignora bordas;
    - aplica threshold;
    - usa apenas o maior componente conectado;
    - remove regiões muito pequenas.
    """
    width, height = original_size

    heatmap_resized = cv2.resize(heatmap, (width, height)).astype(np.float32)
    heatmap_resized = cv2.GaussianBlur(heatmap_resized, (0, 0), sigmaX=8.0)

    max_value = float(np.max(heatmap_resized))
    if max_value <= 1e-8:
        return None

    heatmap_resized = heatmap_resized / max_value

    border_x = int(width * ignore_border_ratio)
    border_y = int(height * ignore_border_ratio)
    heatmap_resized[:border_y, :] = 0
    heatmap_resized[-border_y:, :] = 0
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

    area, x, y, bw, bh = max(components, key=lambda item: item[0])
    box = (int(x), int(y), int(x + bw), int(y + bh))
    box = expand_box(box, image_size=(width, height), padding_ratio=0.15)

    return box


def render_interpretability_section(
    model: keras.Model,
    image_for_model: Image.Image,
    final_class: str,
    bbox_threshold: float,
) -> None:
    st.markdown("---")
    st.markdown("## Interpretabilidade visual")

    st.info(
        "A visualização abaixo mostra regiões que influenciaram a decisão do modelo. "
        "Quando Grad-CAM não está disponível para a arquitetura salva, o app usa um "
        "saliency map baseado nos gradientes da imagem. A bounding box é aproximada "
        "e não equivale a segmentação clínica real da lesão."
    )

    input_array = preprocess_image(image_for_model)
    final_class_index = CLASS_NAMES.index(final_class)

    try:
        last_conv_layer_name = find_last_conv_layer(model)
        heatmap_final = make_gradcam_heatmap(
            input_array=input_array,
            model=model,
            last_conv_layer_name=last_conv_layer_name,
            pred_index=final_class_index,
        )
        method_final = "Grad-CAM"
    except Exception:
        heatmap_final = make_saliency_heatmap(
            input_array=input_array,
            model=model,
            pred_index=final_class_index,
        )
        method_final = "Saliency map"

    heatmap_mel = make_saliency_heatmap(
        input_array=input_array,
        model=model,
        pred_index=MEL_INDEX,
    )
    method_mel = "Saliency map específico para melanoma"

    overlay_final = overlay_heatmap_on_image(
        original_image=image_for_model,
        heatmap=heatmap_final,
        alpha=0.40,
    )

    bbox_final = extract_bounding_box_from_heatmap(
        heatmap=heatmap_final,
        original_size=image_for_model.size,
        threshold=bbox_threshold,
    )

    boxed_final = draw_box_on_image(
        image=image_for_model,
        box=bbox_final,
        label=f"Região: {final_class}",
    )

    overlay_mel = overlay_heatmap_on_image(
        original_image=image_for_model,
        heatmap=heatmap_mel,
        alpha=0.40,
    )

    bbox_mel = extract_bounding_box_from_heatmap(
        heatmap=heatmap_mel,
        original_size=image_for_model.size,
        threshold=bbox_threshold,
    )

    boxed_mel = draw_box_on_image(
        image=image_for_model,
        box=bbox_mel,
        label="Região: melanoma",
    )

    col_a, col_b = st.columns(2)

    with col_a:
        st.image(
            overlay_final,
            caption=f"{method_final} da classe final ({final_class})",
            use_container_width=True,
        )

    with col_b:
        st.image(
            boxed_final,
            caption=f"Bounding box aproximada da classe final ({final_class})",
            use_container_width=True,
        )

    col_c, col_d = st.columns(2)

    with col_c:
        st.image(
            overlay_mel,
            caption=method_mel,
            use_container_width=True,
        )

    with col_d:
        st.image(
            boxed_mel,
            caption="Bounding box aproximada para melanoma",
            use_container_width=True,
        )


# ============================================================
# Interface Streamlit
# ============================================================

st.set_page_config(
    page_title="Skin Lesion Triage Support",
    page_icon="🩺",
    layout="wide",
)

st.title("Smartphone-Based Skin Lesion Classification")
st.subheader("Protótipo web de apoio à triagem inicial de lesões de pele")

st.markdown(
    """
Este protótipo utiliza o modelo **v2 melanoma-sensitive**, desenvolvido na Sprint 3,
para classificar imagens de lesões de pele em uma das sete classes do dataset HAM10000.

O objetivo principal é apoiar a **triagem inicial**, com foco em aumentar a sensibilidade
para casos suspeitos de **melanoma**.
"""
)

st.warning(
    """
Este sistema é apenas uma prova de conceito acadêmica. Ele não fornece diagnóstico médico,
não substitui dermatologistas e não deve ser usado como ferramenta clínica definitiva.
"""
)

st.info(
    """
Para imagens de smartphone, recomenda-se enviar uma foto próxima da lesão, bem iluminada,
centralizada, em foco e com o mínimo possível de fundo. O modelo foi treinado originalmente
com imagens dermatoscópicas do HAM10000, portanto imagens externas podem reduzir a
confiabilidade da predição.
"""
)


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.header("Configurações")

    st.markdown("### Caminhos")

    model_path = st.text_input(
        "Caminho do modelo treinado",
        value=str(DEFAULT_MODEL_PATH),
    )

    policy_path_text = st.text_input(
        "Caminho da política de decisão",
        value=str(DEFAULT_POLICY_PATH),
    )

    policy = load_decision_policy(Path(policy_path_text))
    loaded_threshold = float(
        policy.get("melanoma_threshold", DEFAULT_MELANOMA_THRESHOLD)
    )

    st.markdown("### Thresholds de triagem")

    melanoma_threshold = st.slider(
        "Threshold melanoma-sensitive",
        min_value=0.10,
        max_value=0.90,
        value=loaded_threshold,
        step=0.01,
    )

    intermediate_threshold = st.slider(
        "Threshold de risco intermediário",
        min_value=0.05,
        max_value=float(melanoma_threshold) - 0.01,
        value=min(DEFAULT_INTERMEDIATE_THRESHOLD, float(melanoma_threshold) - 0.01),
        step=0.01,
    )

    st.markdown("### Interpretabilidade")

    bbox_threshold = st.slider(
        "Threshold da bounding box aproximada",
        min_value=0.30,
        max_value=0.90,
        value=DEFAULT_BBOX_THRESHOLD,
        step=0.05,
    )

    st.markdown("---")
    st.markdown("### Classes do modelo")

    for class_name in CLASS_NAMES:
        st.write(f"**{class_name}** — {CLASS_DESCRIPTIONS_PT[class_name]}")


# ============================================================
# Upload da imagem
# ============================================================

uploaded_file = st.file_uploader(
    "Envie uma imagem de lesão de pele",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is None:
    st.info("Envie uma imagem para executar a predição.")

else:
    image = Image.open(uploaded_file).convert("RGB")

    render_quality_warnings(image)

    image_for_model, selected_box, selected_mode = render_crop_controls(image)

    st.markdown("---")
    st.markdown("## Predição")

    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.image(
            image_for_model,
            caption=f"Imagem usada pelo modelo: {selected_mode}",
            use_container_width=True,
        )

    with col2:
        try:
            model = load_trained_model(model_path)

            (
                final_class,
                final_prob,
                probs_by_class,
                argmax_class,
                melanoma_prob,
            ) = predict_image(
                model=model,
                image=image_for_model,
                melanoma_threshold=melanoma_threshold,
            )

            risk_level = get_melanoma_risk_level(
                melanoma_prob=melanoma_prob,
                melanoma_threshold=melanoma_threshold,
                intermediate_threshold=intermediate_threshold,
            )

            st.markdown("## Resultado da triagem")

            st.metric(
                label="Classe final",
                value=f"{final_class} — {CLASS_DESCRIPTIONS_PT[final_class]}",
            )

            st.metric(
                label="Probabilidade da classe final",
                value=format_percentage(final_prob),
            )

            st.metric(
                label="Probabilidade de melanoma",
                value=format_percentage(melanoma_prob),
            )

            st.markdown("### Regra melanoma-sensitive")

            if risk_level == "high":
                st.error(
                    f"A probabilidade de melanoma foi "
                    f"{format_percentage(melanoma_prob)}, acima do threshold de "
                    f"{format_percentage(melanoma_threshold)}. "
                    "A classe final foi ajustada para melanoma."
                )
            elif risk_level == "intermediate":
                st.warning(
                    f"A probabilidade de melanoma foi "
                    f"{format_percentage(melanoma_prob)}, abaixo do threshold principal de "
                    f"{format_percentage(melanoma_threshold)}, mas acima do limiar de risco "
                    f"intermediário de {format_percentage(intermediate_threshold)}. "
                    "Em triagem, recomenda-se cautela e avaliação clínica."
                )
            else:
                st.success(
                    f"A probabilidade de melanoma foi "
                    f"{format_percentage(melanoma_prob)}, abaixo do limiar intermediário de "
                    f"{format_percentage(intermediate_threshold)} e do threshold principal de "
                    f"{format_percentage(melanoma_threshold)}."
                )

            st.markdown("### Interpretação")

            st.write(
                get_triage_message(
                    final_class=final_class,
                    melanoma_prob=melanoma_prob,
                    melanoma_threshold=melanoma_threshold,
                    intermediate_threshold=intermediate_threshold,
                )
            )

            st.markdown("### Comparação com argmax normal")

            st.write(
                "A classe por maior probabilidade antes da regra "
                f"melanoma-sensitive era: **{argmax_class} — "
                f"{CLASS_DESCRIPTIONS_PT[argmax_class]}**."
            )

            st.markdown("### Entrada utilizada")

            if selected_mode == "Imagem inteira":
                st.write(
                    "A predição foi feita na imagem inteira. Isso pode ser mais sensível "
                    "a fundo, iluminação e regiões fora da lesão."
                )
            else:
                st.write(
                    f"A predição foi feita usando: **{selected_mode}**. "
                    "Isso ajuda a reduzir a influência de fundo e regiões não relevantes."
                )

        except Exception as exc:
            st.error("Não foi possível executar a predição.")
            st.exception(exc)
            st.stop()

    render_class_probabilities(probs_by_class)

    try:
        render_interpretability_section(
            model=model,
            image_for_model=image_for_model,
            final_class=final_class,
            bbox_threshold=bbox_threshold,
        )
    except Exception as exc:
        st.warning("Não foi possível gerar a interpretabilidade visual para esta imagem.")
        st.exception(exc)


# ============================================================
# Observação metodológica
# ============================================================

st.markdown("---")

st.markdown(
    """
## Observação metodológica

O modelo foi treinado inicialmente com o dataset **HAM10000**, composto por imagens
dermatoscópicas. Portanto, embora o projeto tenha como visão futura o uso com imagens
de smartphone, esta versão ainda deve ser interpretada como uma **prova de conceito**
e precisa de validação externa antes de qualquer aplicação clínica real.

Nesta versão, o app inclui três melhorias para reduzir limitações observadas nos testes:

1. **Política de risco intermediário para melanoma**: imagens com probabilidade de
melanoma abaixo do threshold principal, mas acima de um limiar intermediário, recebem
um aviso de cautela.

2. **Recorte automático ou manual da lesão**: o usuário pode reduzir a influência de
fundo, iluminação e regiões não relacionadas à lesão antes da classificação.

3. **Interpretabilidade visual com Grad-CAM ou saliency map**: o sistema mostra regiões
que influenciaram a decisão, mas essa visualização não equivale a segmentação clínica
real.

A saída do sistema deve ser interpretada apenas como apoio à triagem inicial.
"""
)
