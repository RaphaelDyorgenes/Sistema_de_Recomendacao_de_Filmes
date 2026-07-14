# Tech Challenge — Fase 02 · Sistema de Recomendação

Sistema de recomendação de filmes baseado no histórico de avaliações dos
usuários (MovieLens-20M). Uma rede neural (NCF — Neural Collaborative
Filtering, em PyTorch) é comparada a baselines de Scikit-Learn, com
pipeline reprodutível em DVC, experimentos e Model Registry no MLflow.

Documentação do modelo: [Model Card](docs/MODEL_CARD.md).

## Estrutura do projeto

```
src/recsys/
  data/           # download, preprocessamento, feature engineering, Dataset PyTorch
  models/         # NCF, baselines (popularidade, KNN de usuarios) e ModelFactory
  preprocessing/  # estrategias de encoding (padrao Strategy)
  training/       # Trainer (early stopping), treino, experimentos e Model Registry
  evaluation/     # metricas de ranking e avaliacao comparativa
  utils/          # configuracoes centralizadas (Pydantic Settings)
tests/            # testes unitarios (pytest)
docs/             # Model Card
data/, models/    # artefatos versionados via DVC (fora do git)
configs/          # configuracoes externas
```

## Requisitos

- Python 3.11 ou superior
- [Poetry](https://python-poetry.org/) para gerenciamento de dependências
- Docker e Docker Compose (opcional, para rodar o pipeline em container)

## Instalação

```bash
# 1. Instalar o Poetry (caso ainda não esteja instalado).
# Nota para Windows: se o comando 'poetry' não for reconhecido, use 'python -m poetry'
pip install poetry

# 2. Na raiz do repositório, instalar as dependências
poetry install

# 3. Criar o arquivo de configuração local
cp .env.example .env        # Windows: copy .env.example .env

# 4. Validar o ambiente
poetry run python scripts/validate_env.py
```

## Obtenção dos dados

O projeto usa o MovieLens-20M, com 20 milhões de avaliações de filmes
(avaliação >= 3.5 é tratada como "o usuário gostou do filme"). O
download é feito via kagglehub:

```bash
poetry run python -m recsys.data.download_data
```

Os CSVs ficam em `data/raw/` (cerca de 700 MB para o `rating.csv`).

## Executar o pipeline (DVC)

```bash
poetry run dvc repro
```

O DVC executa apenas os estágios cujas dependências mudaram:

| Estágio | O que faz | Saídas principais |
|---|---|---|
| `preprocess` | binariza, amostra usuários ativos, encoding, split temporal | `data/processed/{train,val,test}.csv` |
| `feature_eng` | gera negativos (4:1) para treino e validação | `train_with_negatives.csv`, `val_with_negatives.csv` |
| `train` | treina o NCF com early stopping e loga no MLflow | `models/ncf_model.pt` |
| `evaluate` | compara NCF vs baselines com 4 métricas de ranking | `models/evaluation_report.json` |

Para rodar um estágio específico: `poetry run dvc repro train`.

## Experimentos e Model Registry (MLflow)

Depois do `dvc repro`, os experimentos treinam o NCF com três
configurações (embeddings 16, 32 e 64) — cada uma vira um run no MLflow
com parâmetros, métricas e artefatos:

```bash
poetry run python -m recsys.training.experiments
```

O melhor run (menor loss de validação) fica em `models/best_run.json`.
Para registrá-lo no Model Registry e promovê-lo pelos estágios
Staging -> Production:

```bash
poetry run python -m recsys.training.registry
```

Para explorar os runs (aba *Experiments*) e as versões do modelo
`ncf-recommender` (aba *Models*):

```bash
poetry run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Resultados

Métricas de ranking no conjunto de teste (K = 10), sobre 1.174 usuários
ativos do MovieLens-20M:

| Métrica | NCF | Popularidade | KNN de usuários |
|---|---|---|---|
| Precision@10 | **0.2852** | 0.2780 | 0.2796 |
| Recall@10 | 0.0443 | **0.0452** | 0.0451 |
| MAP@10 | 0.1682 | 0.1893 | **0.1911** |
| NDCG@10 | 0.2694 | 0.2927 | **0.2939** |

O NCF vence em precisão do top-10; os baselines ainda ordenam melhor
(MAP/NDCG) neste volume de dados. Análise completa, limitações e vieses
no [Model Card](docs/MODEL_CARD.md).

## Docker

O `docker-compose.yml` sobe um servidor MLflow e roda o treino em
container apontando para ele:

```bash
docker compose up --build
```

- MLflow UI: http://localhost:5000
- `data/` e `models/` são montados como volumes — rodar o download dos
  dados e o `dvc repro` até `feature_eng` no host antes.

## Testes e qualidade

```bash
poetry run pytest             # testes unitarios
poetry run ruff check .       # lint
poetry run ruff format .      # formatacao
poetry run pre-commit install # hooks de lint no commit
```

## Reproduzir os resultados do zero

```bash
poetry install
poetry run python -m recsys.data.download_data
poetry run dvc repro
poetry run python -m recsys.training.experiments
poetry run python -m recsys.training.registry
```

A reprodutibilidade é garantida por sementes fixas (`RANDOM_SEED=42`,
aplicada a PyTorch e NumPy), dados versionados com DVC e todos os runs
rastreados no MLflow.

## 👥 Autores

| Nome                                | Função no Projeto                               |
| :---------------------------------- | :---------------------------------------------- |
| **Mateus de Souza Nascimento**      | Analyst / DevOps / Data Scientist / ML Engineer |
| **Raphael Dyorgenes Vitor**         | Analyst / DevOps / Data Scientist / ML Engineer |
