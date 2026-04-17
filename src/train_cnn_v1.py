from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras


@dataclass
class TrainConfig:
    processed_csv: Path
    output_dir: Path
    image_size: Tuple[int, int]
    batch_size: int
    epochs: int
    learning_rate: float
    seed: int
    melanoma_boost: float


class HAMSequence(keras.utils.Sequence):
    def __init__(
        self,
        data: pd.DataFrame,
        class_count: int,
        image_size: Tuple[int, int],
        batch_size: int,
        shuffle: bool,
    ) -> None:
        self.data = data.reset_index(drop=True)
        self.class_count = class_count
        self.image_size = image_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.data))
        self.on_epoch_end()

    def __len__(self) -> int:
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        start = idx * self.batch_size
        end = min((idx + 1) * self.batch_size, len(self.data))
        batch_indices = self.indices[start:end]
        batch_df = self.data.iloc[batch_indices]

        images = np.stack(
            [
                load_and_preprocess_image(path, self.image_size)
                for path in batch_df["image_path"].tolist()
            ]
        )

        labels_int = batch_df["label_id"].to_numpy(dtype=np.int32)
        labels = keras.utils.to_categorical(labels_int, num_classes=self.class_count)
        return images, labels

    def on_epoch_end(self) -> None:
        if self.shuffle:
            np.random.shuffle(self.indices)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treinar CNN baseline (v1.0) no HAM10000")
    parser.add_argument(
        "--processed-csv",
        type=Path,
        default=Path("data/processed/ham10000_processed.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("reports/model_v1"))
    parser.add_argument("--image-size", type=int, nargs=2, default=(128, 128))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--melanoma-boost",
        type=float,
        default=1.5,
        help="Multiplicador extra no peso da classe melanoma para reduzir falsos negativos.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_and_preprocess_image(path: str, image_size: Tuple[int, int]) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    img = img.resize(image_size)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def validate_columns(df: pd.DataFrame) -> None:
    required = {"image_path", "dx", "label_id", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV processado sem colunas obrigatorias: {sorted(missing)}")


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


def build_model(input_shape: Tuple[int, int, int], num_classes: int, learning_rate: float) -> keras.Model:
    model = keras.Sequential(
        [
            keras.layers.Input(shape=input_shape),
            keras.layers.Conv2D(32, kernel_size=3, padding="same", activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D(pool_size=2),
            keras.layers.Conv2D(64, kernel_size=3, padding="same", activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D(pool_size=2),
            keras.layers.Conv2D(128, kernel_size=3, padding="same", activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D(pool_size=2),
            keras.layers.Flatten(),
            keras.layers.Dense(256, activation="relu"),
            keras.layers.Dropout(0.4),
            keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=[
            keras.metrics.Recall(name="recall"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.CategoricalAccuracy(name="accuracy"),
        ],
    )
    return model


def compute_balanced_weights(
    y_train: np.ndarray,
    class_names: List[str],
    melanoma_boost: float,
) -> Dict[int, float]:
    classes = np.arange(len(class_names))
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    weight_map = {int(c): float(w) for c, w in zip(classes, weights)}

    if "mel" in class_names:
        mel_idx = class_names.index("mel")
        weight_map[mel_idx] *= melanoma_boost

    return weight_map


def save_history_plot(history: keras.callbacks.History, output_dir: Path) -> None:
    hist = pd.DataFrame(history.history)
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(hist["loss"], label="train")
    plt.plot(hist["val_loss"], label="val")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.legend()

    plt.subplot(1, 3, 2)
    if "recall" in hist and "val_recall" in hist:
        plt.plot(hist["recall"], label="train")
        plt.plot(hist["val_recall"], label="val")
    plt.title("Recall")
    plt.xlabel("Epoch")
    plt.legend()

    plt.subplot(1, 3, 3)
    if "accuracy" in hist and "val_accuracy" in hist:
        plt.plot(hist["accuracy"], label="train")
        plt.plot(hist["val_accuracy"], label="val")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "training_history_v1.png", dpi=200)
    plt.close()


def save_confusion_matrix(cm: np.ndarray, class_names: List[str], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_title("Matriz de confusao - Teste (v1)")
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
    fig.savefig(output_dir / "confusion_matrix_v1.png", dpi=200)
    plt.close(fig)


def evaluate_model(
    model: keras.Model,
    test_df: pd.DataFrame,
    class_names: List[str],
    image_size: Tuple[int, int],
    batch_size: int,
    output_dir: Path,
) -> Dict[str, float]:
    test_gen = HAMSequence(
        data=test_df,
        class_count=len(class_names),
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
    )

    y_true = test_df["label_id"].to_numpy(dtype=np.int32)
    y_true_oh = keras.utils.to_categorical(y_true, num_classes=len(class_names))
    y_prob = model.predict(test_gen, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    metrics = {
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

    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(output_dir / "classification_report_v1.csv", index=True)

    cm = confusion_matrix(y_true, y_pred)
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
        output_dir / "confusion_matrix_v1.csv", index=True
    )
    save_confusion_matrix(cm, class_names, output_dir)

    (output_dir / "metrics_v1.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    args = parse_args()
    config = TrainConfig(
        processed_csv=args.processed_csv,
        output_dir=args.output_dir,
        image_size=tuple(args.image_size),
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        melanoma_boost=args.melanoma_boost,
    )

    set_seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(config.processed_csv)
    validate_columns(df)
    train_df, val_df, test_df = split_dataframe(df)
    class_names = get_class_names(df)

    train_gen = HAMSequence(
        data=train_df,
        class_count=len(class_names),
        image_size=config.image_size,
        batch_size=config.batch_size,
        shuffle=True,
    )
    val_gen = HAMSequence(
        data=val_df,
        class_count=len(class_names),
        image_size=config.image_size,
        batch_size=config.batch_size,
        shuffle=False,
    )

    model = build_model(
        input_shape=(config.image_size[0], config.image_size[1], 3),
        num_classes=len(class_names),
        learning_rate=config.learning_rate,
    )

    class_weights = compute_balanced_weights(
        y_train=train_df["label_id"].to_numpy(dtype=np.int32),
        class_names=class_names,
        melanoma_boost=config.melanoma_boost,
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            mode="min",
            restore_best_weights=True,
        )
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config.epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    save_history_plot(history, config.output_dir)

    metrics = evaluate_model(
        model=model,
        test_df=test_df,
        class_names=class_names,
        image_size=config.image_size,
        batch_size=config.batch_size,
        output_dir=config.output_dir,
    )

    model.save(config.output_dir / "cnn_ham10000_v1.keras")

    print("Treinamento concluido com sucesso.")
    print(f"Modelo salvo em: {config.output_dir / 'cnn_ham10000_v1.keras'}")
    print(f"Metricas salvas em: {config.output_dir / 'metrics_v1.json'}")
    print("Resumo de metricas principais:")
    print(f"Recall macro: {metrics['recall_macro']:.4f}")
    print(f"Recall weighted: {metrics['recall_weighted']:.4f}")
    if "recall_melanoma" in metrics:
        print(f"Recall melanoma: {metrics['recall_melanoma']:.4f}")
    print(f"Precision macro: {metrics['precision_macro']:.4f}")
    print(f"F1 macro: {metrics['f1_macro']:.4f}")
    print(f"ROC-AUC OVR macro: {metrics['roc_auc_ovr_macro']:.4f}")


if __name__ == "__main__":
    main()
