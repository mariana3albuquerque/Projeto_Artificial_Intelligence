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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train HAM10000 model v2 focused on melanoma-sensitive triage"
    )

    parser.add_argument(
        "--processed-csv",
        type=Path,
        default=Path("data/processed/ham10000_processed.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/model_v2_mel_sensitive"),
    )

    parser.add_argument("--image-size", type=int, nargs=2, default=(224, 224))
    parser.add_argument("--batch-size", type=int, default=32)

    parser.add_argument("--head-epochs", type=int, default=8)
    parser.add_argument("--finetune-epochs", type=int, default=8)

    parser.add_argument("--learning-rate-head", type=float, default=1e-3)
    parser.add_argument("--learning-rate-finetune", type=float, default=1e-4)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--melanoma-boost",
        type=float,
        default=1.25,
        help="Peso extra moderado para melanoma.",
    )

    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.04,
        help="Suavização dos rótulos para reduzir excesso de confiança.",
    )

    parser.add_argument(
        "--fine-tune-layers",
        type=int,
        default=45,
        help="Número de camadas finais da EfficientNetB0 liberadas no fine-tuning.",
    )

    parser.add_argument(
        "--target-melanoma-recall",
        type=float,
        default=0.72,
        help="Recall desejado para melanoma durante a busca do threshold na validação.",
    )

    parser.add_argument(
        "--min-melanoma-precision",
        type=float,
        default=0.30,
        help="Precision mínima aceitável para melanoma durante a busca do threshold.",
    )

    return parser.parse_args()


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
    """
    O HAM10000 possui múltiplas imagens para algumas lesões.
    Se o CSV tiver lesion_id, verificamos se a mesma lesão aparece em mais de um split.
    Isso é importante para evitar vazamento entre treino, validação e teste.
    """
    if "lesion_id" not in df.columns:
        print(
            "Aviso: coluna lesion_id não encontrada. "
            "Não foi possível verificar vazamento de lesões entre splits."
        )
        return

    split_count = df.groupby("lesion_id")["split"].nunique()
    leaking = split_count[split_count > 1]

    if len(leaking) > 0:
        print(
            f"Aviso: {len(leaking)} lesion_id aparecem em mais de um split. "
            "Isso pode inflar métricas. Idealmente, o split deve ser agrupado por lesion_id."
        )
    else:
        print("Verificação OK: nenhum lesion_id aparece em mais de um split.")


def split_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("Split train/val/test incompleto. Verifique o CSV da Sprint 1.")

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

        def _map_fn(path, label, weight):
            image = decode_and_resize_image(path, image_size)
            label_oh = tf.one_hot(label, depth=num_classes)
            return image, label_oh, weight

        ds = ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)

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


def build_augmentation() -> keras.Sequential:
    return keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.08),
            keras.layers.RandomZoom(0.10),
            keras.layers.RandomTranslation(0.05, 0.05),
            keras.layers.RandomContrast(0.10),
        ],
        name="augmentation",
    )


def build_model(
    input_shape: Tuple[int, int, int],
    num_classes: int,
    learning_rate: float,
    label_smoothing: float,
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
    x = keras.layers.Dropout(0.35)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="ham10000_efficientnetb0_v2_melanoma_sensitive",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
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
            patience=4,
            mode="min",
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]


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

        rows.append(
            {
                "threshold": float(threshold),
                "score": float(score),
                "recall_melanoma": float(recall_mel),
                "precision_melanoma": float(precision_mel),
                "f1_melanoma": float(metrics.get("f1_melanoma", 0.0)),
                "recall_macro": float(recall_macro_value),
                "f1_macro": float(f1_macro_value),
                "accuracy": float(accuracy_value),
            }
        )

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
    fig.savefig(output_dir / "training_history_v2.png", dpi=200)
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


def save_analysis(
    output_dir: Path,
    argmax_metrics: Dict[str, float],
    tuned_metrics: Dict[str, float],
    threshold: float,
) -> None:
    text = f"""# Model v2 - Melanoma-sensitive Analysis

## Objective

The objective of this version is to improve melanoma sensitivity while keeping the classifier reasonably balanced across the seven HAM10000 classes.

## Decision policy

A melanoma-sensitive threshold was selected using the validation set.

- Selected melanoma threshold: {threshold:.2f}

If the predicted probability for melanoma is greater than or equal to this threshold, the final prediction is forced to melanoma.

## Test metrics without melanoma threshold

- Accuracy: {argmax_metrics['accuracy']:.4f}
- Recall macro: {argmax_metrics['recall_macro']:.4f}
- F1 macro: {argmax_metrics['f1_macro']:.4f}
- ROC-AUC OVR macro: {argmax_metrics['roc_auc_ovr_macro']:.4f}
- Recall melanoma: {argmax_metrics.get('recall_melanoma', float('nan')):.4f}
- Precision melanoma: {argmax_metrics.get('precision_melanoma', float('nan')):.4f}
- F1 melanoma: {argmax_metrics.get('f1_melanoma', float('nan')):.4f}

## Test metrics with melanoma-sensitive threshold

- Accuracy: {tuned_metrics['accuracy']:.4f}
- Recall macro: {tuned_metrics['recall_macro']:.4f}
- F1 macro: {tuned_metrics['f1_macro']:.4f}
- ROC-AUC OVR macro: {tuned_metrics['roc_auc_ovr_macro']:.4f}
- Recall melanoma: {tuned_metrics.get('recall_melanoma', float('nan')):.4f}
- Precision melanoma: {tuned_metrics.get('precision_melanoma', float('nan')):.4f}
- F1 melanoma: {tuned_metrics.get('f1_melanoma', float('nan')):.4f}

## Interpretation

This version uses EfficientNetB0 transfer learning, data augmentation, balanced sample weights, moderate melanoma weighting, label smoothing, two-phase training, and a melanoma-sensitive decision threshold.

The threshold strategy is aligned with a triage scenario: increasing the chance of flagging melanoma cases while avoiding the extreme behavior of predicting melanoma for nearly everything.
"""
    (output_dir / "model_analysis_v2.md").write_text(text, encoding="utf-8")


def save_comparison_with_v1(output_dir: Path, tuned_metrics: Dict[str, float]) -> None:
    v1_path = Path("reports/model_v1/metrics_v1.json")

    if not v1_path.exists():
        return

    metrics_v1 = json.loads(v1_path.read_text(encoding="utf-8"))

    keys = [
        "recall_macro",
        "f1_macro",
        "roc_auc_ovr_macro",
        "recall_melanoma",
    ]

    lines = ["# Model Comparison: v1 vs melanoma-sensitive v2", ""]
    lines.append("| Metric | v1 | v2 tuned |")
    lines.append("|---|---:|---:|")

    for key in keys:
        v1 = metrics_v1.get(key, float("nan"))
        v2 = tuned_metrics.get(key, float("nan"))
        lines.append(f"| {key} | {v1:.4f} | {v2:.4f} |")

    lines.append("")
    lines.append("## Summary")
    lines.append(
        "The melanoma-sensitive v2 aims to increase melanoma recall while preserving multiclass balance."
    )

    (output_dir / "model_comparison.md").write_text("\n".join(lines), encoding="utf-8")


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
    )

    set_seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(config.processed_csv)

    validate_columns(df)
    warn_if_lesion_leakage(df)

    train_df, val_df, test_df = split_dataframe(df)
    class_names = get_class_names(df)

    print("Classes:", class_names)
    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")

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
    )

    history_head = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.head_epochs,
        callbacks=get_callbacks(config.output_dir, "head"),
        verbose=1,
    )

    unfreeze_backbone(backbone, config.fine_tune_layers)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate_finetune),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=config.label_smoothing),
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
    history_df.to_csv(config.output_dir / "history_v2.csv", index=False)
    save_history_plot(history_df, config.output_dir)

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
        "image_size": list(config.image_size),
        "architecture": "EfficientNetB0",
        "task": "melanoma-sensitive triage",
    }

    (config.output_dir / "decision_policy_v2.json").write_text(
        json.dumps(decision_policy, indent=2),
        encoding="utf-8",
    )

    argmax_metrics = evaluate_and_save(
        model=model,
        df_eval=test_df,
        class_names=class_names,
        image_size=config.image_size,
        batch_size=config.batch_size,
        output_dir=config.output_dir,
        prefix="v2_argmax",
        melanoma_threshold=None,
    )

    tuned_metrics = evaluate_and_save(
        model=model,
        df_eval=test_df,
        class_names=class_names,
        image_size=config.image_size,
        batch_size=config.batch_size,
        output_dir=config.output_dir,
        prefix="v2",
        melanoma_threshold=melanoma_threshold,
    )

    model.save(config.output_dir / "cnn_ham10000_v2.keras")

    save_analysis(
        output_dir=config.output_dir,
        argmax_metrics=argmax_metrics,
        tuned_metrics=tuned_metrics,
        threshold=melanoma_threshold,
    )

    save_comparison_with_v1(config.output_dir, tuned_metrics)

    print("Treinamento v2 melanoma-sensitive concluído.")
    print(f"Modelo salvo em: {config.output_dir / 'cnn_ham10000_v2.keras'}")
    print(f"Threshold melanoma escolhido: {melanoma_threshold:.2f}")
    print("Métricas finais com threshold melanoma-sensitive:")
    print(json.dumps(tuned_metrics, indent=2))


if __name__ == "__main__":
    main()