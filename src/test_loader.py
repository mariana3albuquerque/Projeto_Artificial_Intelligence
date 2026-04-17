from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test initial HAM10000 loader")
    parser.add_argument("--processed-csv", type=Path, required=True)
    parser.add_argument("--image-size", type=int, nargs=2, default=(224, 224))
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def load_and_preprocess_image(path: str, image_size: Tuple[int, int]) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    img = img.resize(image_size)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.processed_csv)
    batch_df = df[df["split"] == "train"].head(args.batch_size)
    if batch_df.empty:
        raise ValueError("Nenhuma imagem de treino encontrada no CSV processado.")

    images = np.stack([
        load_and_preprocess_image(path, tuple(args.image_size))
        for path in batch_df["image_path"].tolist()
    ])
    labels = batch_df["label_id"].to_numpy()

    print("Batch carregado com sucesso.")
    print(f"images.shape = {images.shape}")
    print(f"labels.shape = {labels.shape}")
    print(f"classes no batch = {batch_df['dx'].tolist()}")


if __name__ == "__main__":
    main()
