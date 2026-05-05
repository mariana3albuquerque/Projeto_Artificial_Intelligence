from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow import keras


# ============================================================
# Configurações principais
# ============================================================

DEFAULT_MODEL_PATH = Path("reports/model_v2_mel_sensitive/cnn_ham10000_v2.keras")
DEFAULT_POLICY_PATH = Path("reports/model_v2_mel_sensitive/decision_policy_v2.json")

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

DEFAULT_MELANOMA_THRESHOLD = 0.36


# ============================================================
# Funções auxiliares
# ============================================================

@st.cache_resource
def load_model(model_path: str) -> keras.Model:
    """
    Carrega o modelo treinado da Sprint 3.
    O cache evita recarregar o modelo a cada interação da interface.
    """
    path = Path(model_path)

    if not path.exists():
        raise FileNotFoundError(f"Modelo não encontrado em: {path}")

    model = keras.models.load_model(path)
    return model


def load_decision_policy(policy_path: Path) -> Dict[str, float]:
    """
    Carrega a política de decisão, incluindo o threshold de melanoma.
    Caso o arquivo não exista, usa o threshold padrão.
    """
    if not policy_path.exists():
        return {"melanoma_threshold": DEFAULT_MELANOMA_THRESHOLD}

    try:
        with policy_path.open("r", encoding="utf-8") as file:
            policy = json.load(file)
        return policy
    except Exception:
        return {"melanoma_threshold": DEFAULT_MELANOMA_THRESHOLD}


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Pré-processa a imagem enviada pelo usuário.

    Importante:
    O modelo v2 foi treinado com imagens redimensionadas para 224x224.
    A normalização/preprocessamento específico da EfficientNet já está dentro
    do próprio modelo, caso o modelo tenha sido salvo com essa camada.
    Por isso, aqui mantemos os pixels em escala 0-255 como float32.
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
    """
    Retorna:
    - classe final após aplicar threshold melanoma-sensitive;
    - probabilidade da classe final;
    - probabilidades por classe;
    - classe original pelo argmax;
    - probabilidade de melanoma.
    """
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


def get_risk_message(final_class: str, melanoma_prob: float, threshold: float) -> str:
    if final_class == "mel":
        return (
            "A imagem foi marcada como suspeita para melanoma pela regra "
            "melanoma-sensitive. Recomenda-se avaliação médica especializada."
        )

    if melanoma_prob >= threshold * 0.75:
        return (
            "A probabilidade de melanoma não ultrapassou o threshold final, "
            "mas ficou relativamente próxima. Recomenda-se cautela na triagem."
        )

    return (
        "A imagem não foi marcada como melanoma pela regra de triagem. "
        "Mesmo assim, o resultado não substitui avaliação médica."
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

with st.sidebar:
    st.header("Configurações")

    model_path = st.text_input(
        "Caminho do modelo treinado",
        value=str(DEFAULT_MODEL_PATH),
    )

    policy_path_text = st.text_input(
        "Caminho da política de decisão",
        value=str(DEFAULT_POLICY_PATH),
    )

    policy = load_decision_policy(Path(policy_path_text))
    loaded_threshold = float(policy.get("melanoma_threshold", DEFAULT_MELANOMA_THRESHOLD))

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
        st.write(f"**{class_name}** — {CLASS_DESCRIPTIONS[class_name]}")

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
        st.image(image, caption="Imagem enviada", use_container_width=True)

    with col2:
        try:
            model = load_model(model_path)

            final_class, final_prob, probs_by_class, argmax_class, melanoma_prob = predict_image(
                model=model,
                image=image,
                melanoma_threshold=melanoma_threshold,
            )

            st.markdown("## Resultado da triagem")

            st.metric(
                label="Classe final",
                value=f"{final_class} — {CLASS_DESCRIPTIONS[final_class]}",
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
                    f"A probabilidade de melanoma foi {format_percentage(melanoma_prob)}, "
                    f"acima do threshold de {format_percentage(melanoma_threshold)}. "
                    "A classe final foi ajustada para melanoma."
                )
            else:
                st.success(
                    f"A probabilidade de melanoma foi {format_percentage(melanoma_prob)}, "
                    f"abaixo do threshold de {format_percentage(melanoma_threshold)}."
                )

            st.markdown("### Interpretação")
            st.write(get_risk_message(final_class, melanoma_prob, melanoma_threshold))

            st.markdown("### Comparação com argmax normal")
            st.write(
                f"A classe por maior probabilidade antes da regra melanoma-sensitive era: "
                f"**{argmax_class} — {CLASS_DESCRIPTIONS[argmax_class]}**."
            )

        except Exception as exc:
            st.error("Não foi possível executar a predição.")
            st.exception(exc)

    st.markdown("---")
    st.markdown("## Probabilidades por classe")

    sorted_probs = sorted(
        probs_by_class.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    for class_name, prob in sorted_probs:
        st.progress(prob)
        st.write(
            f"**{class_name}** — {CLASS_DESCRIPTIONS[class_name]}: "
            f"{format_percentage(prob)}"
        )

    st.markdown("---")
    st.markdown("## Top 3 predições")

    top_3 = sorted_probs[:3]

    for rank, (class_name, prob) in enumerate(top_3, start=1):
        st.write(
            f"{rank}. **{class_name}** — {CLASS_DESCRIPTIONS[class_name]} "
            f"({format_percentage(prob)})"
        )

st.markdown("---")

st.markdown(
    """
### Observação metodológica

O modelo foi treinado inicialmente com o dataset HAM10000, composto por imagens
dermatoscópicas. Portanto, embora o projeto tenha como visão futura o uso com imagens
de smartphone, esta versão ainda deve ser interpretada como uma prova de conceito
e precisa de validação externa antes de qualquer aplicação clínica real.
"""
)