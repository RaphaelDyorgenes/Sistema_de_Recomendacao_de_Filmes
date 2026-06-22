"""Métricas de avaliação para sistemas de recomendação.

Implementa as 4 métricas exigidas pelo Tech Challenge:
- Precision@K
- Recall@K
- MAP@K (Mean Average Precision)
- NDCG@K (Normalized Discounted Cumulative Gain)

Todas as funções operam sobre listas de ids, sem dependência de
frameworks, para facilitar testes e reutilização.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def precision_at_k(actual: set[int], predicted: Sequence[int], k: int) -> float:
    """Fração dos itens recomendados que são relevantes.

    Args:
        actual: Conjunto de itens realmente relevantes para o usuário.
        predicted: Lista ordenada de itens recomendados.
        k: Número de recomendações a considerar.

    Returns:
        Precision@K em ``[0.0, 1.0]``.
    """
    if k <= 0:
        return 0.0
    top_k = predicted[:k]
    hits = sum(1 for item in top_k if item in actual)
    return hits / k


def recall_at_k(actual: set[int], predicted: Sequence[int], k: int) -> float:
    """Fração dos itens relevantes que foram recomendados.

    Args:
        actual: Conjunto de itens realmente relevantes.
        predicted: Lista ordenada de itens recomendados.
        k: Número de recomendações a considerar.

    Returns:
        Recall@K em ``[0.0, 1.0]``. Retorna ``0.0`` se não há itens relevantes.
    """
    if not actual or k <= 0:
        return 0.0
    top_k = predicted[:k]
    hits = sum(1 for item in top_k if item in actual)
    return hits / len(actual)


def _average_precision(actual: set[int], predicted: Sequence[int], k: int) -> float:
    """Average Precision para um único usuário.

    Args:
        actual: Itens relevantes.
        predicted: Itens recomendados, em ordem.
        k: Corte.

    Returns:
        AP@K para este usuário.
    """
    if not actual or k <= 0:
        return 0.0

    top_k = predicted[:k]
    score = 0.0
    hits = 0

    for i, item in enumerate(top_k):
        if item in actual:
            hits += 1
            score += hits / (i + 1)

    return score / min(len(actual), k)


def map_at_k(
    actuals: Sequence[set[int]],
    predictions: Sequence[Sequence[int]],
    k: int,
) -> float:
    """Mean Average Precision@K sobre múltiplos usuários.

    Args:
        actuals: Lista de conjuntos de itens relevantes por usuário.
        predictions: Lista de listas ordenadas de recomendações por usuário.
        k: Número de recomendações a considerar.

    Returns:
        MAP@K médio sobre todos os usuários.
    """
    if not actuals:
        return 0.0
    total = sum(
        _average_precision(act, pred, k)
        for act, pred in zip(actuals, predictions, strict=True)
    )
    return total / len(actuals)


def _dcg_at_k(actual: set[int], predicted: Sequence[int], k: int) -> float:
    """Discounted Cumulative Gain para um único usuário.

    Args:
        actual: Itens relevantes.
        predicted: Itens recomendados, em ordem.
        k: Corte.

    Returns:
        DCG@K (relevância binária).
    """
    top_k = predicted[:k]
    return sum(1.0 / math.log2(i + 2) for i, item in enumerate(top_k) if item in actual)


def ndcg_at_k(actual: set[int], predicted: Sequence[int], k: int) -> float:
    """Normalized Discounted Cumulative Gain@K.

    Mede a qualidade do ranking: um item relevante no topo vale mais
    do que no final da lista. Normalizado pelo DCG ideal.

    Args:
        actual: Conjunto de itens realmente relevantes.
        predicted: Lista ordenada de itens recomendados.
        k: Número de recomendações a considerar.

    Returns:
        NDCG@K em ``[0.0, 1.0]``. Retorna ``0.0`` se não há itens relevantes.
    """
    if not actual or k <= 0:
        return 0.0

    dcg = _dcg_at_k(actual, predicted, k)

    # DCG ideal: todos os relevantes nas primeiras posições
    ideal_predicted = list(actual)[:k]
    idcg = _dcg_at_k(actual, ideal_predicted, k)

    if idcg == 0:
        return 0.0
    return dcg / idcg
