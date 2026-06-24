"""Pré-processamento dos dados brutos do MovieLens para o pipeline de recomendação.

Etapas executadas (stage ``preprocess`` do DVC):
1. Leitura do ``rating.csv`` bruto
2. Amostragem determinística (se configurado)
3. Binarização: rating >= 3.5 → interação positiva (simula navegação/clique)
4. Label-encoding de user_id e item_id para índices contíguos
5. Split temporal em treino / validação / teste (64/16/20)
6. Persistência dos artefatos processados
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import pandas as pd

from recsys.preprocessing.strategies import LabelEncodeStrategy
from recsys.utils.config import settings

# Limiar para binarização: simula "usuário se interessou pelo produto".
_RATING_THRESHOLD = 3.5

# Proporções do split temporal.
_TRAIN_FRAC = 0.64
_VAL_FRAC = 0.16


def run(
    raw_dir: Path | None = None,
    processed_dir: Path | None = None,
    sample_size: int | None = None,
    seed: int | None = None,
) -> None:
    """Executa o pipeline de pré-processamento completo.

    Args:
        raw_dir: Diretório com ``rating.csv``. Padrão: ``settings.data_raw_dir``.
        processed_dir: Diretório de saída. Padrão: ``settings.data_processed_dir``.
        sample_size: Quantidade de interações a amostrar. 0 = todas.
        seed: Semente para reprodutibilidade.
    """
    raw_dir = raw_dir or settings.data_raw_dir
    processed_dir = processed_dir or settings.data_processed_dir
    sample_size = sample_size if sample_size is not None else settings.data_sample_size
    seed = seed if seed is not None else settings.random_seed

    processed_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Leitura ──────────────────────────────────────────────
    ratings_path = raw_dir / "rating.csv"
    if not ratings_path.exists():
        msg = f"rating.csv não encontrado em {raw_dir}"
        raise FileNotFoundError(msg)

    print(f"📄 Lendo {ratings_path}...")
    df = pd.read_csv(ratings_path)
    print(f"   Interações brutas: {len(df):,}")

    # ── 2. Amostragem ──────────────────────────────────────────
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=seed)
        print(f"   Amostra: {len(df):,} interações (seed={seed})")

    # ── 3. Binarização ──────────────────────────────────────────
    df = df[df["rating"] >= _RATING_THRESHOLD].copy()
    df["label"] = 1
    print(f"   Interações positivas (rating >= {_RATING_THRESHOLD}): {len(df):,}")

    # ── 4. Label Encoding ───────────────────────────────────────
    user_encoder = LabelEncodeStrategy()
    item_encoder = LabelEncodeStrategy()

    user_encoder.fit(df["userId"].tolist())
    item_encoder.fit(df["movieId"].tolist())

    df["user_id"] = user_encoder.transform(df["userId"].tolist())
    df["item_id"] = item_encoder.transform(df["movieId"].tolist())

    n_users = user_encoder.vocabulary_size
    n_items = item_encoder.vocabulary_size
    print(f"   Usuários: {n_users:,} | Itens: {n_items:,}")

    # ── 5. Split temporal ───────────────────────────────────────
    df = df.sort_values("timestamp")

    n = len(df)
    train_end = int(n * _TRAIN_FRAC)
    val_end = int(n * (_TRAIN_FRAC + _VAL_FRAC))

    cols = ["user_id", "item_id", "label"]
    train_df = df.iloc[:train_end][cols]
    val_df = df.iloc[train_end:val_end][cols]
    test_df = df.iloc[val_end:][cols]

    print(
        f"   Split: treino={len(train_df):,} | "
        f"val={len(val_df):,} | teste={len(test_df):,}"
    )

    # ── 6. Persistência ─────────────────────────────────────────
    train_df.to_csv(processed_dir / "train.csv", index=False)
    val_df.to_csv(processed_dir / "val.csv", index=False)
    test_df.to_csv(processed_dir / "test.csv", index=False)

    # Encoders para inverse-transform futuro
    with (processed_dir / "encoders.pkl").open("wb") as f:
        pickle.dump({"user": user_encoder, "item": item_encoder}, f)

    # Metadados para o modelo
    metadata = {"n_users": n_users, "n_items": n_items, "seed": seed}
    with (processed_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("✅ Pré-processamento concluído!")


if __name__ == "__main__":
    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    sys.exit(run())  # type: ignore[func-returns-value]
