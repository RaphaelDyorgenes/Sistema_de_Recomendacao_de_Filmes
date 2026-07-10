"""Feature engineering: geração de amostras negativas para treino e validação.

Em recomendação implícita, o modelo precisa aprender a distinguir itens
com os quais o usuário interagiu (positivos) de itens que ele **não**
interagiu (negativos). Este script gera pares negativos aleatórios para
o treino e para a validação — sem negativos, a loss de validação seria
calculada só sobre positivos e o early stopping perderia o sentido.

Stage ``feature_eng`` do DVC.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from recsys.utils.config import settings

# Proporção negativo:positivo por par.
_NEG_RATIO = 4


def _positives_by_user(df: pd.DataFrame) -> dict[int, set[int]]:
    """Agrupa os itens positivos de cada usuário.

    Args:
        df: Interações com colunas ``user_id`` e ``item_id``.

    Returns:
        Dicionário user_id → {item_ids positivos}.
    """
    positives: dict[int, set[int]] = {}
    for uid, iid in zip(df["user_id"], df["item_id"], strict=True):
        positives.setdefault(uid, set()).add(iid)
    return positives


def _generate_negatives(
    positives_by_user: dict[int, set[int]],
    known_by_user: dict[int, set[int]],
    n_items: int,
    neg_ratio: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Sorteia pares negativos por usuário, evitando itens conhecidos.

    Args:
        positives_by_user: Positivos do split sendo aumentado.
        known_by_user: Todos os positivos do usuário (treino + validação),
            para não sortear um falso negativo.
        n_items: Tamanho do vocabulário de itens.
        neg_ratio: Quantos negativos gerar por positivo.
        rng: Gerador de números aleatórios com semente fixa.

    Returns:
        DataFrame com colunas ``user_id``, ``item_id`` e ``label`` = 0.
    """
    all_items = set(range(n_items))
    users: list[int] = []
    items: list[int] = []

    for uid, positives in positives_by_user.items():
        candidates = list(all_items - known_by_user[uid])
        n_neg = min(len(positives) * neg_ratio, len(candidates))
        if n_neg == 0:
            continue
        sampled = rng.choice(candidates, size=n_neg, replace=False)
        users.extend([uid] * n_neg)
        items.extend(sampled.tolist())

    return pd.DataFrame({"user_id": users, "item_id": items, "label": 0})


def _augment_split(
    df: pd.DataFrame,
    known_by_user: dict[int, set[int]],
    n_items: int,
    neg_ratio: int,
    rng: np.random.Generator,
    seed: int,
    out_path: Path,
) -> int:
    """Concatena positivos e negativos de um split e persiste embaralhado.

    Args:
        df: Positivos do split.
        known_by_user: Positivos conhecidos do usuário em todos os splits.
        n_items: Tamanho do vocabulário de itens.
        neg_ratio: Quantos negativos gerar por positivo.
        rng: Gerador de números aleatórios.
        seed: Semente para o embaralhamento final.
        out_path: CSV de saída.

    Returns:
        Quantidade de negativos gerados.
    """
    neg_df = _generate_negatives(
        _positives_by_user(df), known_by_user, n_items, neg_ratio, rng
    )
    full_df = pd.concat([df, neg_df], ignore_index=True)
    full_df = full_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    full_df.to_csv(out_path, index=False)
    print(f"   {out_path.name}: {len(df):,} positivos + {len(neg_df):,} negativos")
    return len(neg_df)


def run(
    processed_dir: Path | None = None,
    neg_ratio: int = _NEG_RATIO,
    seed: int | None = None,
) -> None:
    """Gera amostras negativas para os conjuntos de treino e validação.

    Args:
        processed_dir: Diretório com ``train.csv``, ``val.csv`` e metadados.
        neg_ratio: Quantos negativos gerar por positivo.
        seed: Semente para reprodutibilidade.
    """
    processed_dir = processed_dir or settings.data_processed_dir
    seed = seed if seed is not None else settings.random_seed
    rng = np.random.default_rng(seed)

    train_df = pd.read_csv(processed_dir / "train.csv")
    val_df = pd.read_csv(processed_dir / "val.csv")
    with (processed_dir / "metadata.json").open(encoding="utf-8") as f:
        meta = json.load(f)

    # Positivos de treino + validação: nenhum vira falso negativo.
    known = _positives_by_user(pd.concat([train_df, val_df], ignore_index=True))

    n_items = meta["n_items"]
    n_neg_train = _augment_split(
        train_df,
        known,
        n_items,
        neg_ratio,
        rng,
        seed,
        processed_dir / "train_with_negatives.csv",
    )
    n_neg_val = _augment_split(
        val_df,
        known,
        n_items,
        neg_ratio,
        rng,
        seed,
        processed_dir / "val_with_negatives.csv",
    )

    meta.update(
        {
            "n_train_positives": len(train_df),
            "n_train_negatives": n_neg_train,
            "n_val_positives": len(val_df),
            "n_val_negatives": n_neg_val,
            "neg_ratio": neg_ratio,
        }
    )
    with (processed_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("✅ Feature engineering concluído!")


if __name__ == "__main__":
    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    sys.exit(run())  # type: ignore[func-returns-value]
