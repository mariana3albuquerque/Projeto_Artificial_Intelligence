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
    / "model_v3_mel_focal"
    / "cnn_ham10000_v3.keras"
)

DEFAULT_POLICY_PATH = (
    PROJECT_ROOT
    / "reports"
    / "model_v3_mel_focal"
    / "decision_policy_v3.json"
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
MEL_INDEX = CLASS_NAMES.index("mel")


# ============================================================
# Funções auxiliares
# ============================================================

@st.cache_resource
def load_trained_model(model_path: str) -> keras.Model:
    path = Path(model_path)

    if not path.exists():
        raise FileNotFoundError(f"Modelo não encontrado em: {path}")

    # compile=False evita erro ao carregar modelos com loss customizada (focal loss v3)
    return keras.models.load_model(path, compile=False)


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


def preprocess_image(image: Image.Image) -> np.ndarray:
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


def format_percentage(value: float) -> str:
    return f"{value * 100:.2f}%"


def get_triage_message(
    final_class: str,
    melanoma_prob: float,
    melanoma_threshold: float,
) -> str:
    if final_class == "mel":
        return (
            "A imagem foi marcada como suspeita para melanoma pela regra "
            "melanoma-sensitive. Recomenda-se avaliação médica especializada."
        )

    if melanoma_prob >= melanoma_threshold * 0.75:
        return (
            "A probabilidade de melanoma não ultrapassou o threshold final, "
            "mas ficou relativamente próxima. Recomenda-se cautela na triagem."
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
# Grad-CAM e bounding box aproximada
# ============================================================

def find_last_conv_layer(
    model: keras.Model,
) -> Tuple[Optional[keras.Model], str]:
    """
    Encontra a última camada convolucional 4D.
    Busca primeiro nas camadas top-level; se não achar, desce para backbones
    aninhados (ex.: EfficientNetB0 com pooling='avg' que absorve o GAP interno).
    Retorna (backbone_layer_or_None, layer_name).
    """
    for layer in reversed(model.layers):
        try:
            if len(layer.output.shape) == 4:
                return None, layer.name
        except Exception:
            continue

    for layer in reversed(model.layers):
        if not isinstance(layer, keras.Model):
            continue
        for sub_layer in reversed(layer.layers):
            try:
                if len(sub_layer.output.shape) == 4:
                    return layer, sub_layer.name
            except Exception:
                continue

    raise ValueError(
        "Não foi possível encontrar automaticamente uma camada convolucional 4D."
    )

def _build_gradcam_model(
    model: keras.Model,
    backbone: Optional[keras.Model],
    last_conv_layer_name: str,
) -> keras.Model:
    """
    Constrói um modelo que emite [conv_activation, predicao_final].

    O problema com backbone aninhado (EfficientNetB0 com pooling='avg'):
    backbone.get_layer(name).output retorna um tensor do grafo INTERNO do
    backbone, não acessível a partir de model.inputs. A solução é reconstruir
    o modelo camada a camada, substituindo o backbone por uma versão dual que
    expõe simultaneamente o tensor conv e o output pooled.
    """
    if backbone is None:
        return tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv_layer_name).output, model.output],
        )

    backbone_dual = tf.keras.models.Model(
        inputs=backbone.inputs,
        outputs=[
            backbone.get_layer(last_conv_layer_name).output,
            backbone.output,
        ],
    )

    new_input = keras.Input(shape=model.input.shape[1:])
    x = new_input
    conv_tensor = None

    for layer in model.layers:
        if isinstance(layer, keras.layers.InputLayer):
            continue
        if layer.name == backbone.name:
            conv_tensor, x = backbone_dual(x, training=False)
        else:
            try:
                x = layer(x, training=False)
            except TypeError:
                x = layer(x)

    return keras.models.Model(inputs=new_input, outputs=[conv_tensor, x])


def make_gradcam_heatmap(
    input_array: np.ndarray,
    model: keras.Model,
    last_conv_layer_name: str,
    backbone: Optional[keras.Model] = None,
    pred_index: Optional[int] = None,
) -> np.ndarray:
    """
    Gera Grad-CAM. Levanta ValueError se não conseguir, para o chamador
    usar saliency map como fallback.
    """
    try:
        grad_model = _build_gradcam_model(model, backbone, last_conv_layer_name)

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(input_array, training=False)

            if pred_index is None:
                pred_index = tf.argmax(predictions[0])

            class_channel = predictions[:, pred_index]

        grads = tape.gradient(class_channel, conv_outputs)

        if grads is None:
            raise ValueError("Gradientes nulos para Grad-CAM.")

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0)
        max_value = tf.reduce_max(heatmap)

        if max_value == 0:
            raise ValueError("Heatmap Grad-CAM vazio.")

        return (heatmap / max_value).numpy()

    except Exception as exc:
        raise ValueError(
            "Não foi possível gerar Grad-CAM com a arquitetura salva. "
            "Use saliency map como fallback."
        ) from exc
    
def make_saliency_heatmap(
    input_array: np.ndarray,
    model: keras.Model,
    pred_index: Optional[int] = None,
) -> np.ndarray:
    """
    Gera um mapa de saliência baseado no gradiente da classe em relação à imagem.
    Funciona mesmo quando Grad-CAM não consegue acessar uma camada convolucional.
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
    """
    Sobrepõe o heatmap na imagem original.
    """
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
    threshold: float = 0.50,
    padding_pct: float = 0.03,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Extrai bounding box a partir da região mais ativada do heatmap.
    Aplica blur gaussiano antes do threshold para reduzir ruído, usa o maior
    contorno para ignorar ativações espúrias, e adiciona padding proporcional.
    """
    width, height = original_size

    heatmap_resized = cv2.resize(heatmap, (width, height))

    sigma = max(width, height) * 0.02
    blurred = cv2.GaussianBlur(heatmap_resized, (0, 0), sigmaX=sigma)
    h_min, h_max = blurred.min(), blurred.max()
    if h_max - h_min > 1e-8:
        blurred = (blurred - h_min) / (h_max - h_min)

    mask = (blurred >= threshold).astype(np.uint8)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    pad_x = int(width * padding_pct)
    pad_y = int(height * padding_pct)

    x_min = max(0, x - pad_x)
    y_min = max(0, y - pad_y)
    x_max = min(width, x + w + pad_x)
    y_max = min(height, y + h + pad_y)

    return int(x_min), int(y_min), int(x_max), int(y_max)


def draw_bounding_box(
    image: Image.Image,
    bbox: Optional[Tuple[int, int, int, int]],
    label: str,
) -> Image.Image:
    image = image.convert("RGB").copy()

    if bbox is None:
        return image

    draw = ImageDraw.Draw(image)
    x_min, y_min, x_max, y_max = bbox

    draw.rectangle(
        [x_min, y_min, x_max, y_max],
        outline="red",
        width=5,
    )

    text_y = max(0, y_min - 24)
    draw.text((x_min, text_y), label, fill="red")

    return image


def render_gradcam_section(
    model: keras.Model,
    image: Image.Image,
    final_class: str,
) -> None:
    """
    Renderiza Grad-CAM quando possível. Se Grad-CAM falhar, usa saliency map.
    Também gera bounding box aproximada a partir do mapa de calor.
    """
    st.markdown("---")
    st.markdown("## Interpretabilidade visual")

    st.info(
        "A visualização abaixo mostra regiões que influenciaram a decisão do modelo. "
        "Quando Grad-CAM não está disponível para a arquitetura salva, o app usa "
        "um saliency map baseado em gradientes da imagem. A bounding box é aproximada "
        "e não equivale a segmentação clínica real da lesão."
    )

    input_array = preprocess_image(image)
    final_class_index = CLASS_NAMES.index(final_class)

    try:
        backbone, last_conv_layer_name = find_last_conv_layer(model)

        heatmap_final = make_gradcam_heatmap(
            input_array=input_array,
            model=model,
            last_conv_layer_name=last_conv_layer_name,
            backbone=backbone,
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

    try:
        heatmap_mel = make_saliency_heatmap(
            input_array=input_array,
            model=model,
            pred_index=MEL_INDEX,
        )

        method_mel = "Saliency map específico para melanoma"

    except Exception as exc:
        st.warning("Não foi possível gerar mapa de interpretabilidade para melanoma.")
        st.exception(exc)
        return

    overlay_final = overlay_heatmap_on_image(
        original_image=image,
        heatmap=heatmap_final,
        alpha=0.40,
    )

    bbox_final = extract_bounding_box_from_heatmap(
        heatmap=heatmap_final,
        original_size=image.size,
        threshold=0.60,
    )

    boxed_final = draw_bounding_box(
        image=image,
        bbox=bbox_final,
        label=f"Região: {final_class}",
    )

    overlay_mel = overlay_heatmap_on_image(
        original_image=image,
        heatmap=heatmap_mel,
        alpha=0.40,
    )

    bbox_mel = extract_bounding_box_from_heatmap(
        heatmap=heatmap_mel,
        original_size=image.size,
        threshold=0.60,
    )

    boxed_mel = draw_bounding_box(
        image=image,
        bbox=bbox_mel,
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

    st.markdown("### Threshold")

    melanoma_threshold = st.slider(
        "Threshold melanoma-sensitive",
        min_value=0.10,
        max_value=0.90,
        value=loaded_threshold,
        step=0.01,
    )

    st.markdown("---")
    st.markdown("### Classes do modelo")

    for class_name in CLASS_NAMES:
        st.write(
            f"**{class_name}** — {CLASS_DESCRIPTIONS_PT[class_name]}"
        )


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
    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.image(
            image,
            caption="Imagem enviada",
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
                image=image,
                melanoma_threshold=melanoma_threshold,
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

            if melanoma_prob >= melanoma_threshold:
                st.error(
                    f"A probabilidade de melanoma foi "
                    f"{format_percentage(melanoma_prob)}, acima do threshold de "
                    f"{format_percentage(melanoma_threshold)}. "
                    "A classe final foi ajustada para melanoma."
                )
            else:
                st.success(
                    f"A probabilidade de melanoma foi "
                    f"{format_percentage(melanoma_prob)}, abaixo do threshold de "
                    f"{format_percentage(melanoma_threshold)}."
                )

            st.markdown("### Interpretação")

            st.write(
                get_triage_message(
                    final_class=final_class,
                    melanoma_prob=melanoma_prob,
                    melanoma_threshold=melanoma_threshold,
                )
            )

            st.markdown("### Comparação com argmax normal")

            st.write(
                "A classe por maior probabilidade antes da regra "
                f"melanoma-sensitive era: **{argmax_class} — "
                f"{CLASS_DESCRIPTIONS_PT[argmax_class]}**."
            )

            render_class_probabilities(probs_by_class)

            render_gradcam_section(
                model=model,
                image=image,
                final_class=final_class,
            )

        except Exception as exc:
            st.error("Não foi possível executar a predição.")
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

O modelo atual realiza **classificação da imagem inteira**. A visualização Grad-CAM
ajuda a indicar quais regiões influenciaram a decisão, mas não substitui uma etapa de
segmentação clínica da lesão.

A saída do sistema deve ser interpretada apenas como apoio à triagem inicial.
"""
)