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

Isso valida se o pipeline de carregamento e pré-processamento está funcionando.

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
