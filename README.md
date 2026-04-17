# Sprint 1 - HAM10000 Preparation

Este pacote implementa a Sprint 1 do projeto **Smartphone-Based Skin Lesion Classification for Early Triage Support**.

## O que está incluído

- organização e validação da base HAM10000
- documentação das 7 classes
- análise da distribuição das classes
- split estratificado em treino, validação e teste
- pipeline inicial de carregamento e pré-processamento

## Estrutura esperada da base

Coloque os arquivos do Kaggle dentro de `data/raw/` de uma destas formas:

### Opção A

```text
/data/raw/
  HAM10000_metadata.csv
  HAM10000_images_part_1/
  HAM10000_images_part_2/
```

### Opção B

```text
/data/raw/
  HAM10000_metadata.csv
  images/
```

## Instalação

```bash
pip install -r requirements.txt
```

## Baixar a base HAM10000 (Kaggle)

1. Crie o token de API no Kaggle em **Account > API > Create New Token**.
2. Salve o arquivo `kaggle.json` em `C:/Users/<seu_usuario>/.kaggle/kaggle.json`.
3. Na raiz do projeto, execute:

```bash
.venv/Scripts/python.exe -m pip install kaggle
.venv/Scripts/python.exe -m kaggle.cli datasets download -d kmader/skin-cancer-mnist-ham10000 -p data/raw --unzip
```

Após o download, a pasta `data/raw` deve conter o metadata e as imagens do HAM10000.

## Etapa 1 — Preparar a base

```bash
python src/prepare_ham10000.py \
  --raw-dir data/raw \
  --processed-dir data/processed \
  --reports-dir reports \
  --train-size 0.70 \
  --val-size 0.15 \
  --test-size 0.15 \
  --seed 42
```

Saídas esperadas:
- `data/processed/ham10000_processed.csv`
- `reports/class_distribution.csv`
- `reports/class_distribution.png`
- `reports/sample_images.png`
- `reports/dataset_description.md`
- `reports/sprint1_summary.md`

## Etapa 2 — Testar o pipeline inicial

```bash
python src/test_loader.py \
  --processed-csv data/processed/ham10000_processed.csv \
  --image-size 224 224 \
  --batch-size 8
```

No PowerShell (Windows), use:

```powershell
python src/test_loader.py `
  --processed-csv data/processed/ham10000_processed.csv `
  --image-size 224 224 `
  --batch-size 8
```

Isso valida se o pipeline de carregamento e pré-processamento está funcionando.

## Etapa 3 — Treinar modelo de classificacao (Pipeline 1.0)

O baseline da Sprint 2 utiliza uma CNN simples com:
- camada de entrada para pixels RGB
- camadas convolucionais com kernels para extracao de caracteristicas
- camada de saida com 7 nos (uma por classe do HAM10000)

Comando:

```bash
python src/train_cnn_v1.py \
  --processed-csv data/processed/ham10000_processed.csv \
  --output-dir reports/model_v1 \
  --image-size 128 128 \
  --batch-size 32 \
  --epochs 25 \
  --learning-rate 0.001 \
  --seed 42 \
  --melanoma-boost 1.5
```

No PowerShell (Windows), use:

```powershell
python src/train_cnn_v1.py `
  --processed-csv data/processed/ham10000_processed.csv `
  --output-dir reports/model_v1 `
  --image-size 128 128 `
  --batch-size 32 `
  --epochs 25 `
  --learning-rate 0.001 `
  --seed 42 `
  --melanoma-boost 1.5
```

Tambem funciona em uma unica linha no PowerShell:

```powershell
python src/train_cnn_v1.py --processed-csv data/processed/ham10000_processed.csv --output-dir reports/model_v1 --image-size 128 128 --batch-size 32 --epochs 25 --learning-rate 0.001 --seed 42 --melanoma-boost 1.5
```

O script reaproveita o split estratificado criado na Sprint 1 (train/val/test).

Saidas esperadas em `reports/model_v1/`:
- `cnn_ham10000_v1.keras` (modelo treinado)
- `metrics_v1.json` (metricas agregadas)
- `classification_report_v1.csv` (precision/recall/f1 por classe)
- `confusion_matrix_v1.csv` (matriz numerica)
- `confusion_matrix_v1.png` (matriz visual)
- `training_history_v1.png` (curvas de treino)
- `training_history_compact_v1.png` (grafico compacto com loss e accuracy)

## Metricas criticas de avaliacao

Prioridade clinica:
- Recall (sensibilidade), com destaque para recall da classe melanoma (`mel`)

Metricas complementares:
- Precision
- F1-score
- ROC-AUC (multiclasse OVR)

Analise qualitativa:
- Matriz de confusao para identificar classes mais confundidas

## Classes do dataset

- `akiec` — Actinic keratoses / intraepithelial carcinoma
- `bcc` — Basal cell carcinoma
- `bkl` — Benign keratosis-like lesions
- `df` — Dermatofibroma
- `mel` — Melanoma
- `nv` — Melanocytic nevi
- `vasc` — Vascular lesions

## Observações

- O foco desta sprint é deixar a base pronta para treinar o modelo baseline.
- O script faz split estratificado por classe.
- O projeto continua sendo uma prova de conceito. O HAM10000 é um benchmark inicial, mas não equivale a fotos reais de smartphone.
