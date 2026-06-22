"""Feature engineering: geração de amostras negativas para treinamento.

Em recomendação implícita, o modelo precisa aprender a distinguir itens
com os quais o usuário interagiu (positivos) de itens que ele **não**
interagiu (negativos). Este script gera pares negativos aleatórios
mantendo a proporção configurável.

Stage ``feature_eng`` do DVC.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from recsys.utils.config import settings

# Proporção negativo:positivo por par de treino.
_NEG_RATIO = 4


def run(
    processed_dir: Path | None = None,
    neg_ratio: int = _NEG_RATIO,
    seed: int | None = None,
) -> None:
    """Gera amostras negativas e concatena ao conjunto de treino.

    Args:
        processed_dir: Diretório com ``train.csv`` e ``metadata.json``.
        neg_ratio: Quantos negativos gerar por positivo.
        seed: Semente para reprodutibilidade.
    """
    processed_dir = processed_dir or settings.data_processed_dir
    seed = seed if seed is not None else settings.random_seed

    rng = np.random.default_rng(seed)

    # ── Carregar dados ──────────────────────────────────────────
    train_df = pd.read_csv(processed_dir / "train.csv")
    with (processed_dir / "metadata.json").open(encoding="utf-8") as f:
        meta = json.load(f)

    n_items = meta["n_items"]
    all_items = set(range(n_items))

    # ── Histórico por usuário ───────────────────────────────────
    user_positives: dict[int, set[int]] = {}
    for uid, iid in zip(train_df["user_id"], train_df["item_id"], strict=True):
        user_positives.setdefault(uid, set()).add(iid)

    # ── Gerar negativos ─────────────────────────────────────────
    neg_users: list[int] = []
    neg_items: list[int] = []
    neg_labels: list[int] = []

    for uid, positives in user_positives.items():
        negatives = list(all_items - positives)
        n_neg = min(len(positives) * neg_ratio, len(negatives))
        if n_neg == 0:
            continue
        sampled = rng.choice(negatives, size=n_neg, replace=False)
        neg_users.extend([uid] * n_neg)
        neg_items.extend(sampled.tolist())
        neg_labels.extend([0] * n_neg)

    neg_df = pd.DataFrame(
        {
            "user_id": neg_users,
            "item_id": neg_items,
            "label": neg_labels,
        }
    )

    # ── Concatenar e embaralhar ─────────────────────────────────
    full_df = pd.concat([train_df, neg_df], ignore_index=True)
    full_df = full_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    out_path = processed_dir / "train_with_negatives.csv"
    full_df.to_csv(out_path, index=False)

    # Atualizar metadata
    meta["n_train_positives"] = len(train_df)
    meta["n_train_negatives"] = len(neg_df)
    meta["neg_ratio"] = neg_ratio
    with (processed_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("✅ Feature engineering concluído!")
    print(f"   Positivos: {len(train_df):,} | Negativos: {len(neg_df):,}")
    print(f"   Total treino: {len(full_df):,} → {out_path}")


if __name__ == "__main__":
    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    sys.exit(run())  # type: ignore[func-returns-value]
