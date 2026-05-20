from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_class_weight
import keras


# ============================================================
# Focal Loss multiclasse (softmax)
# ============================================================

class SparseCategoricalFocalLoss(keras.losses.Loss):
    """
    Focal loss para classificação multiclasse com rótulos one-hot e saída softmax.
    Penaliza mais os exemplos difíceis (baixa confiança), o que ajuda especialmente
    em classes minoritárias como melanoma.
    Referência: Lin et al. 2017 - Focal Loss for Dense Object Detection.
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
# Config
# ============================================================

@dataclass
class TrainConfig:
    processed_csv: Path
    output_dir: Path
    image_size: Tuple[int, int]
    batch_size: int
    head_epochs: int
    finetune_epochs: int
    learning_rate_head: float
    learning_rate_finetune: float
    seed: int
    melanoma_boost: float
    label_smoothing: float
    fine_tune_layers: int
    target_melanoma_recall: float
    min_melanoma_precision: float
    use_focal_loss: bool
    focal_gamma: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train HAM10000 model v3 — improved melanoma sensitivity"
    )

    parser.add_argument(
        "--processed-csv",
        type=Path,
        default=Path("data/processed/ham10000_processed.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/model_v3_mel_focal"),
    )

    parser.add_argument("--image-size", type=int, nargs=2, default=(224, 224))
    parser.add_argument("--batch-size", type=int, default=32)

    parser.add_argument("--head-epochs", type=int, default=10)
    parser.add_argument("--finetune-epochs", type=int, default=12)

    parser.add_argument("--learning-rate-head", type=float, default=1e-3)
    parser.add_argument("--learning-rate-finetune", type=float, default=1e-4)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--melanoma-boost",
        type=float,
        default=2.5,
        help="Peso extra para melanoma. v2 usava 1.25; aumentar para 2.5 força o "
             "modelo a errar menos em melanoma.",
    )

    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.01,
        help="Suavização de rótulos. v2 usava 0.04, o que suprimia a confiança do "
             "modelo em melanoma. Valor menor preserva confiança.",
    )

    parser.add_argument(
        "--fine-tune-layers",
        type=int,
        default=60,
        help="Camadas finais da EfficientNetB0 liberadas no fine-tuning. "
             "v2 usava 45; aumentar para 60 dá mais expressividade.",
    )

    parser.add_argument(
        "--target-melanoma-recall",
        type=float,
        default=0.80,
        help="Recall alvo para melanoma na busca de threshold. v2 usava 0.72.",
    )

    parser.add_argument(
        "--min-melanoma-precision",
        type=float,
        default=0.25,
        help="Precision mínima aceitável para melanoma. Reduzida levemente para "
             "permitir maior recall.",
    )

    parser.add_argument(
        "--use-focal-loss",
        action="store_true",
        default=True,
        help="Usar focal loss ao invés de cross-entropy. Recomendado para "
             "classes desbalanceadas como melanoma.",
    )

    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
        help="Expoente gamma da focal loss. Valores maiores focam mais nos "
             "exemplos difíceis.",
    )

    return parser.parse_args()


# ============================================================
# Funções de dados (idênticas ao v2)
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def validate_columns(df: pd.DataFrame) -> None:
    required = {"image_path", "dx", "label_id", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV processado sem colunas obrigatórias: {sorted(missing)}")


def warn_if_lesion_leakage(df: pd.DataFrame) -> None:
    if "lesion_id" not in df.columns:
        print("Aviso: coluna lesion_id não encontrada.")
        return
    split_count = df.groupby("lesion_id")["split"].nunique()
    leaking = split_count[split_count > 1]
    if len(leaking) > 0:
        print(f"Aviso: {len(leaking)} lesion_id aparecem em mais de um split.")
    else:
        print("Verificação OK: nenhum lesion_id aparece em mais de um split.")


def split_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()
    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("Split train/val/test incompleto.")
    return train_df, val_df, test_df


def get_class_names(df: pd.DataFrame) -> List[str]:
    class_map = (
        df[["label_id", "dx"]]
        .drop_duplicates()
        .sort_values("label_id")
        .reset_index(drop=True)
    )
    return class_map["dx"].tolist()


def compute_class_weights(
    y_train: np.ndarray,
    class_names: List[str],
    melanoma_boost: float,
) -> Dict[int, float]:
    classes = np.arange(len(class_names))
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )
    weight_map = {int(c): float(w) for c, w in zip(classes, weights)}
    if "mel" in class_names:
        mel_idx = class_names.index("mel")
        weight_map[mel_idx] *= melanoma_boost
    return weight_map


def add_sample_weights(df: pd.DataFrame, class_weights: Dict[int, float]) -> pd.DataFrame:
    df = df.copy()
    df["sample_weight"] = df["label_id"].map(class_weights).astype(np.float32)
    return df


def decode_and_resize_image(path: tf.Tensor, image_size: Tuple[int, int]) -> tf.Tensor:
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, image_size)
    image = tf.cast(image, tf.float32)
    return image


def build_dataset(
    df: pd.DataFrame,
    num_classes: int,
    image_size: Tuple[int, int],
    batch_size: int,
    training: bool,
    include_sample_weights: bool = False,
) -> tf.data.Dataset:
    paths = df["image_path"].astype(str).to_numpy()
    labels = df["label_id"].astype(np.int32).to_numpy()

    if include_sample_weights:
        weights = df["sample_weight"].astype(np.float32).to_numpy()
        ds = tf.data.Dataset.from_tensor_slices((paths, labels, weights))

        def _map_fn_w(path, label, weight):
            image = decode_and_resize_image(path, image_size)
            label_oh = tf.one_hot(label, depth=num_classes)
            return image, label_oh, weight

        ds = ds.map(_map_fn_w, num_parallel_calls=tf.data.AUTOTUNE)

    else:
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))

        def _map_fn(path, label):
            image = decode_and_resize_image(path, image_size)
            label_oh = tf.one_hot(label, depth=num_classes)
            return image, label_oh

        ds = ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        ds = ds.shuffle(buffer_size=min(len(df), 2048), reshuffle_each_iteration=True)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ============================================================
# Augmentation mais agressiva para v3
# ============================================================

def build_augmentation() -> keras.Sequential:
    return keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal_and_vertical"),
            keras.layers.RandomRotation(0.15),
            keras.layers.RandomZoom(0.15),
            keras.layers.RandomTranslation(0.08, 0.08),
            keras.layers.RandomContrast(0.20),
            keras.layers.RandomBrightness(0.15),
        ],
        name="augmentation",
    )


# ============================================================
# Modelo
# ============================================================

def build_model(
    input_shape: Tuple[int, int, int],
    num_classes: int,
    learning_rate: float,
    label_smoothing: float,
    use_focal_loss: bool,
    focal_gamma: float,
) -> Tuple[keras.Model, keras.Model]:
    data_augmentation = build_augmentation()

    try:
        backbone = keras.applications.EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
            pooling="avg",
        )
        print("EfficientNetB0 carregada com pesos ImageNet.")
    except Exception as exc:
        print(f"Não foi possível carregar pesos ImageNet. Usando weights=None. Erro: {exc}")
        backbone = keras.applications.EfficientNetB0(
            include_top=False,
            weights=None,
            input_shape=input_shape,
            pooling="avg",
        )

    backbone.trainable = False

    inputs = keras.Input(shape=input_shape)
    x = data_augmentation(inputs)
    x = keras.applications.efficientnet.preprocess_input(x)
    x = backbone(x, training=False)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dropout(0.40)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="ham10000_efficientnetb0_v3_mel_focal",
    )

    if use_focal_loss:
        loss_fn = SparseCategoricalFocalLoss(
            gamma=focal_gamma,
            label_smoothing=label_smoothing,
        )
        print(f"Usando focal loss (gamma={focal_gamma}, label_smoothing={label_smoothing})")
    else:
        loss_fn = keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)
        print(f"Usando cross-entropy (label_smoothing={label_smoothing})")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_fn,
        metrics=[keras.metrics.CategoricalAccuracy(name="accuracy")],
    )

    return model, backbone


def unfreeze_backbone(backbone: keras.Model, fine_tune_layers: int) -> None:
    backbone.trainable = True
    for layer in backbone.layers[:-fine_tune_layers]:
        layer.trainable = False
    for layer in backbone.layers:
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False


def get_callbacks(output_dir: Path, phase_name: str) -> List[keras.callbacks.Callback]:
    return [
        keras.callbacks.ModelCheckpoint(
            filepath=output_dir / f"best_{phase_name}.keras",
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            mode="min",
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]


# ============================================================
# Avaliação e métricas (idênticas ao v2)
# ============================================================

def merge_histories(
    history_head: keras.callbacks.History,
    history_ft: keras.callbacks.History,
) -> pd.DataFrame:
    df1 = pd.DataFrame(history_head.history)
    df1["phase"] = "head"
    df2 = pd.DataFrame(history_ft.history)
    df2["phase"] = "finetune"
    return pd.concat([df1, df2], ignore_index=True)


def predict_with_melanoma_threshold(
    y_prob: np.ndarray,
    class_names: List[str],
    melanoma_threshold: Optional[float],
) -> np.ndarray:
    y_pred = np.argmax(y_prob, axis=1)
    if melanoma_threshold is None or "mel" not in class_names:
        return y_pred
    mel_idx = class_names.index("mel")
    melanoma_mask = y_prob[:, mel_idx] >= melanoma_threshold
    y_pred[melanoma_mask] = mel_idx
    return y_pred


def compute_metrics_from_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
) -> Dict[str, float]:
    y_true_oh = keras.utils.to_categorical(y_true, num_classes=len(class_names))

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "roc_auc_ovr_macro": float(
            roc_auc_score(y_true_oh, y_prob, average="macro", multi_class="ovr")
        ),
        "roc_auc_ovr_weighted": float(
            roc_auc_score(y_true_oh, y_prob, average="weighted", multi_class="ovr")
        ),
    }

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    if "mel" in class_names:
        metrics["recall_melanoma"] = float(report["mel"]["recall"])
        metrics["precision_melanoma"] = float(report["mel"]["precision"])
        metrics["f1_melanoma"] = float(report["mel"]["f1-score"])

    return metrics


def select_melanoma_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
    target_melanoma_recall: float,
    min_melanoma_precision: float,
    output_dir: Path,
) -> float:
    if "mel" not in class_names:
        return 1.01

    rows = []

    for threshold in np.round(np.arange(0.10, 0.91, 0.01), 2):
        y_pred = predict_with_melanoma_threshold(
            y_prob=y_prob,
            class_names=class_names,
            melanoma_threshold=float(threshold),
        )
        metrics = compute_metrics_from_predictions(
            y_true=y_true,
            y_prob=y_prob,
            y_pred=y_pred,
            class_names=class_names,
        )

        recall_mel = metrics.get("recall_melanoma", 0.0)
        precision_mel = metrics.get("precision_melanoma", 0.0)
        f1_macro_value = metrics.get("f1_macro", 0.0)
        recall_macro_value = metrics.get("recall_macro", 0.0)
        accuracy_value = metrics.get("accuracy", 0.0)

        penalty_precision = max(0.0, min_melanoma_precision - precision_mel)

        score = (
            0.45 * recall_mel
            + 0.25 * f1_macro_value
            + 0.20 * recall_macro_value
            + 0.10 * accuracy_value
            - 0.45 * penalty_precision
        )

        rows.append({
            "threshold": float(threshold),
            "score": float(score),
            "recall_melanoma": float(recall_mel),
            "precision_melanoma": float(precision_mel),
            "f1_melanoma": float(metrics.get("f1_melanoma", 0.0)),
            "recall_macro": float(recall_macro_value),
            "f1_macro": float(f1_macro_value),
            "accuracy": float(accuracy_value),
        })

    df_thresholds = pd.DataFrame(rows)
    df_thresholds.to_csv(output_dir / "melanoma_threshold_search.csv", index=False)

    candidates = df_thresholds[
        (df_thresholds["recall_melanoma"] >= target_melanoma_recall)
        & (df_thresholds["precision_melanoma"] >= min_melanoma_precision)
    ].copy()

    if not candidates.empty:
        candidates = candidates.sort_values(
            by=["f1_macro", "recall_macro", "precision_melanoma", "accuracy"],
            ascending=False,
        )
        return float(candidates.iloc[0]["threshold"])

    df_thresholds = df_thresholds.sort_values(by="score", ascending=False)
    return float(df_thresholds.iloc[0]["threshold"])


def save_history_plot(history_df: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history_df["loss"], label="train")
    axes[0].plot(history_df["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(history_df["accuracy"], label="train")
    axes[1].plot(history_df["val_accuracy"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "training_history_v3.png", dpi=200)
    plt.close(fig)


def save_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_dir: Path,
    filename: str,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=200)
    plt.close(fig)


def evaluate_and_save(
    model: keras.Model,
    df_eval: pd.DataFrame,
    class_names: List[str],
    image_size: Tuple[int, int],
    batch_size: int,
    output_dir: Path,
    prefix: str,
    melanoma_threshold: Optional[float],
) -> Dict[str, float]:
    eval_ds = build_dataset(
        df=df_eval,
        num_classes=len(class_names),
        image_size=image_size,
        batch_size=batch_size,
        training=False,
        include_sample_weights=False,
    )

    y_true = df_eval["label_id"].to_numpy(dtype=np.int32)
    y_prob = model.predict(eval_ds, verbose=0)

    y_pred = predict_with_melanoma_threshold(
        y_prob=y_prob,
        class_names=class_names,
        melanoma_threshold=melanoma_threshold,
    )

    metrics = compute_metrics_from_predictions(
        y_true=y_true,
        y_prob=y_prob,
        y_pred=y_pred,
        class_names=class_names,
    )

    if melanoma_threshold is not None:
        metrics["melanoma_threshold"] = float(melanoma_threshold)

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    pd.DataFrame(report).transpose().to_csv(
        output_dir / f"classification_report_{prefix}.csv",
        index=True,
    )

    cm = confusion_matrix(y_true, y_pred)

    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
        output_dir / f"confusion_matrix_{prefix}.csv",
        index=True,
    )

    save_confusion_matrix(
        cm=cm,
        class_names=class_names,
        output_dir=output_dir,
        filename=f"confusion_matrix_{prefix}.png",
        title=f"Matriz de confusão - {prefix}",
    )

    (output_dir / f"metrics_{prefix}.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    return metrics


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = parse_args()

    config = TrainConfig(
        processed_csv=args.processed_csv,
        output_dir=args.output_dir,
        image_size=tuple(args.image_size),
        batch_size=args.batch_size,
        head_epochs=args.head_epochs,
        finetune_epochs=args.finetune_epochs,
        learning_rate_head=args.learning_rate_head,
        learning_rate_finetune=args.learning_rate_finetune,
        seed=args.seed,
        melanoma_boost=args.melanoma_boost,
        label_smoothing=args.label_smoothing,
        fine_tune_layers=args.fine_tune_layers,
        target_melanoma_recall=args.target_melanoma_recall,
        min_melanoma_precision=args.min_melanoma_precision,
        use_focal_loss=args.use_focal_loss,
        focal_gamma=args.focal_gamma,
    )

    set_seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(config.processed_csv)
    validate_columns(df)
    warn_if_lesion_leakage(df)

    train_df, val_df, test_df = split_dataframe(df)
    class_names = get_class_names(df)

    print("Classes:", class_names)
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    class_weights = compute_class_weights(
        y_train=train_df["label_id"].to_numpy(dtype=np.int32),
        class_names=class_names,
        melanoma_boost=config.melanoma_boost,
    )
    print("Class weights:", class_weights)

    train_df = add_sample_weights(train_df, class_weights)

    train_ds = build_dataset(
        df=train_df,
        num_classes=len(class_names),
        image_size=config.image_size,
        batch_size=config.batch_size,
        training=True,
        include_sample_weights=True,
    )

    val_ds = build_dataset(
        df=val_df,
        num_classes=len(class_names),
        image_size=config.image_size,
        batch_size=config.batch_size,
        training=False,
        include_sample_weights=False,
    )

    model, backbone = build_model(
        input_shape=(config.image_size[0], config.image_size[1], 3),
        num_classes=len(class_names),
        learning_rate=config.learning_rate_head,
        label_smoothing=config.label_smoothing,
        use_focal_loss=config.use_focal_loss,
        focal_gamma=config.focal_gamma,
    )

    print("\n=== Fase 1: treinando só o head ===")
    history_head = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.head_epochs,
        callbacks=get_callbacks(config.output_dir, "head"),
        verbose=1,
    )

    print(f"\n=== Fase 2: fine-tuning ({config.fine_tune_layers} camadas) ===")
    unfreeze_backbone(backbone, config.fine_tune_layers)

    if config.use_focal_loss:
        loss_fn = SparseCategoricalFocalLoss(
            gamma=config.focal_gamma,
            label_smoothing=config.label_smoothing,
        )
    else:
        loss_fn = keras.losses.CategoricalCrossentropy(
            label_smoothing=config.label_smoothing
        )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate_finetune),
        loss=loss_fn,
        metrics=[keras.metrics.CategoricalAccuracy(name="accuracy")],
    )

    history_finetune = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.finetune_epochs,
        callbacks=get_callbacks(config.output_dir, "finetune"),
        verbose=1,
    )

    history_df = merge_histories(history_head, history_finetune)
    history_df.to_csv(config.output_dir / "history_v3.csv", index=False)
    save_history_plot(history_df, config.output_dir)

    print("\n=== Selecionando threshold de melanoma ===")
    val_prob = model.predict(val_ds, verbose=0)
    val_true = val_df["label_id"].to_numpy(dtype=np.int32)

    melanoma_threshold = select_melanoma_threshold(
        y_true=val_true,
        y_prob=val_prob,
        class_names=class_names,
        target_melanoma_recall=config.target_melanoma_recall,
        min_melanoma_precision=config.min_melanoma_precision,
        output_dir=config.output_dir,
    )

    decision_policy = {
        "melanoma_threshold": melanoma_threshold,
        "target_melanoma_recall": config.target_melanoma_recall,
        "min_melanoma_precision": config.min_melanoma_precision,
        "melanoma_boost": config.melanoma_boost,
        "label_smoothing": config.label_smoothing,
        "fine_tune_layers": config.fine_tune_layers,
        "use_focal_loss": config.use_focal_loss,
        "focal_gamma": config.focal_gamma,
        "image_size": list(config.image_size),
        "architecture": "EfficientNetB0",
        "task": "melanoma-sensitive triage v3",
    }

    (config.output_dir / "decision_policy_v3.json").write_text(
        json.dumps(decision_policy, indent=2),
        encoding="utf-8",
    )

    print("\n=== Avaliação no conjunto de teste ===")
    argmax_metrics = evaluate_and_save(
        model=model,
        df_eval=test_df,
        class_names=class_names,
        image_size=config.image_size,
        batch_size=config.batch_size,
        output_dir=config.output_dir,
        prefix="v3_argmax",
        melanoma_threshold=None,
    )

    tuned_metrics = evaluate_and_save(
        model=model,
        df_eval=test_df,
        class_names=class_names,
        image_size=config.image_size,
        batch_size=config.batch_size,
        output_dir=config.output_dir,
        prefix="v3",
        melanoma_threshold=melanoma_threshold,
    )

    model.save(config.output_dir / "cnn_ham10000_v3.keras")

    print("\n=== Treinamento v3 concluído ===")
    print(f"Modelo salvo em: {config.output_dir / 'cnn_ham10000_v3.keras'}")
    print(f"Threshold melanoma escolhido: {melanoma_threshold:.2f}")
    print("\nMétricas finais (com threshold melanoma-sensitive):")
    print(json.dumps(tuned_metrics, indent=2))

    print("\n=== Comparação v2 → v3 ===")
    v2_path = Path("reports/model_v2_mel_sensitive_final/metrics_v2.json")
    if v2_path.exists():
        v2_metrics = json.loads(v2_path.read_text())
        for key in ["recall_melanoma", "precision_melanoma", "f1_melanoma", "f1_macro", "accuracy"]:
            v2_val = v2_metrics.get(key, float("nan"))
            v3_val = tuned_metrics.get(key, float("nan"))
            delta = v3_val - v2_val
            print(f"  {key}: v2={v2_val:.4f} → v3={v3_val:.4f}  ({delta:+.4f})")


if __name__ == "__main__":
    main()
