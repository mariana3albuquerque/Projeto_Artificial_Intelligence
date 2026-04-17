from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

DX_TO_NAME: Dict[str, str] = {
    "akiec": "Actinic keratoses / intraepithelial carcinoma",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions",
}

@dataclass
class SplitConfig:
    train_size: float
    val_size: float
    test_size: float
    seed: int

    def validate(self) -> None:
        total = self.train_size + self.val_size + self.test_size
        if not np.isclose(total, 1.0):
            raise ValueError(f"train + val + test deve somar 1.0, mas somou {total:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare HAM10000 for Sprint 1")
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--reports-dir", type=Path, required=True)
    parser.add_argument("--train-size", type=float, default=0.70)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-count", type=int, default=9)
    return parser.parse_args()


def find_metadata_csv(raw_dir: Path) -> Path:
    matches = list(raw_dir.rglob("HAM10000_metadata.csv"))
    if not matches:
        raise FileNotFoundError("HAM10000_metadata.csv não encontrado em data/raw")
    return matches[0]


def build_image_index(raw_dir: Path) -> Dict[str, Path]:
    image_paths: List[Path] = []
    for pattern in ("*.jpg", "*.jpeg", "*.png"):
        image_paths.extend(raw_dir.rglob(pattern))

    index: Dict[str, Path] = {}
    for path in image_paths:
        stem = path.stem
        if stem not in index:
            index[stem] = path.resolve()
    return index


def load_metadata(raw_dir: Path) -> pd.DataFrame:
    metadata_path = find_metadata_csv(raw_dir)
    df = pd.read_csv(metadata_path)
    required_columns = {"image_id", "dx"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes no metadata: {sorted(missing)}")
    return df


def attach_image_paths(df: pd.DataFrame, image_index: Dict[str, Path]) -> pd.DataFrame:
    out = df.copy()
    out["image_path"] = out["image_id"].map(lambda x: str(image_index.get(str(x), "")))
    missing = (out["image_path"] == "").sum()
    if missing > 0:
        missing_ids = out.loc[out["image_path"] == "", "image_id"].head(10).tolist()
        raise FileNotFoundError(
            f"{missing} imagens do metadata não foram encontradas. Exemplos: {missing_ids}"
        )
    out["dx_name"] = out["dx"].map(DX_TO_NAME).fillna("Unknown")
    return out


def make_split(df: pd.DataFrame, config: SplitConfig) -> pd.DataFrame:
    config.validate()
    train_df, temp_df = train_test_split(
        df,
        test_size=(config.val_size + config.test_size),
        random_state=config.seed,
        stratify=df["dx"],
    )

    relative_val = config.val_size / (config.val_size + config.test_size)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - relative_val),
        random_state=config.seed,
        stratify=temp_df["dx"],
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    final_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    final_df["label_id"] = pd.Categorical(final_df["dx"], categories=sorted(DX_TO_NAME)).codes
    return final_df


def save_distribution_reports(df: pd.DataFrame, reports_dir: Path) -> pd.DataFrame:
    dist = (
        df.groupby(["dx", "dx_name"]).size().reset_index(name="count").sort_values("count", ascending=False)
    )
    dist["percentage"] = (dist["count"] / dist["count"].sum() * 100).round(2)
    dist.to_csv(reports_dir / "class_distribution.csv", index=False)

    plt.figure(figsize=(10, 5))
    plt.bar(dist["dx"], dist["count"])
    plt.title("Distribuição das classes - HAM10000")
    plt.xlabel("Classe")
    plt.ylabel("Número de imagens")
    plt.tight_layout()
    plt.savefig(reports_dir / "class_distribution.png", dpi=200)
    plt.close()
    return dist


def save_sample_images(df: pd.DataFrame, reports_dir: Path, sample_count: int, seed: int) -> None:
    sample_df = df.sample(n=min(sample_count, len(df)), random_state=seed).reset_index(drop=True)
    cols = 3
    rows = int(np.ceil(len(sample_df) / cols))
    plt.figure(figsize=(10, 10))
    for idx, row in sample_df.iterrows():
        img = Image.open(row["image_path"]).convert("RGB")
        plt.subplot(rows, cols, idx + 1)
        plt.imshow(img)
        plt.title(f"{row['dx']}\n{row['image_id']}")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(reports_dir / "sample_images.png", dpi=200)
    plt.close()


def write_dataset_description(reports_dir: Path) -> None:
    lines = [
        "# Descrição da base HAM10000\n",
        "## Classes\n",
    ]
    for code, name in DX_TO_NAME.items():
        lines.append(f"- **{code}** — {name}\n")
    lines.append(
        "\n## Observação\nO HAM10000 será utilizado como benchmark inicial para desenvolvimento e avaliação do modelo."
    )
    (reports_dir / "dataset_description.md").write_text("".join(lines), encoding="utf-8")


def write_summary(df: pd.DataFrame, dist: pd.DataFrame, reports_dir: Path) -> None:
    split_counts = df["split"].value_counts().to_dict()
    summary = f"""# Sprint 1 - Resumo

## Total de imagens
- {len(df)}

## Split
- treino: {split_counts.get('train', 0)}
- validação: {split_counts.get('val', 0)}
- teste: {split_counts.get('test', 0)}

## Número de classes
- {df['dx'].nunique()}

## Classe mais frequente
- {dist.iloc[0]['dx']} ({dist.iloc[0]['count']} imagens)

## Classe menos frequente
- {dist.iloc[-1]['dx']} ({dist.iloc[-1]['count']} imagens)
"""
    (reports_dir / "sprint1_summary.md").write_text(summary, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.processed_dir.mkdir(parents=True, exist_ok=True)
    args.reports_dir.mkdir(parents=True, exist_ok=True)

    config = SplitConfig(
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
    )

    metadata = load_metadata(args.raw_dir)
    image_index = build_image_index(args.raw_dir)
    prepared = attach_image_paths(metadata, image_index)
    prepared = make_split(prepared, config)

    processed_csv = args.processed_dir / "ham10000_processed.csv"
    prepared.to_csv(processed_csv, index=False)

    dist = save_distribution_reports(prepared, args.reports_dir)
    save_sample_images(prepared, args.reports_dir, args.sample_count, args.seed)
    write_dataset_description(args.reports_dir)
    write_summary(prepared, dist, args.reports_dir)

    print(f"CSV processado salvo em: {processed_csv}")
    print(f"Relatórios salvos em: {args.reports_dir}")


if __name__ == "__main__":
    main()
