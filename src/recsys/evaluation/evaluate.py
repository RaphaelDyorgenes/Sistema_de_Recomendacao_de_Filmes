"""Avaliação comparativa do modelo NCF contra os baselines Scikit-Learn.

Stage ``evaluate`` do DVC. Gera recomendações para cada usuário do
conjunto de teste, calcula 4 métricas de ranking por modelo e loga os
resultados no MLflow para comparação direta.

Os modelos são criados via ``ModelFactory``: adicionar um novo baseline
exige apenas registrá-lo e incluí-lo em ``_BASELINE_NAMES``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mlflow
import pandas as pd
import torch

from recsys.evaluation.metrics import (
    map_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from recsys.models.base import RecommenderModel
from recsys.models.factory import ModelFactory
from recsys.models.ncf import NCFRecommender as _  # noqa: F401 — registra no Factory
from recsys.models.popularity import PopularityRecommender as __  # noqa: F401
from recsys.models.user_knn import UserKNNRecommender as ___  # noqa: F401
from recsys.utils.config import settings

_K = 10

# Baselines Scikit-Learn avaliados contra o modelo neural.
_BASELINE_NAMES = ("popularity", "user_knn")


def _build_ground_truth(
    test_df: pd.DataFrame,
) -> dict[int, set[int]]:
    """Monta o ground truth: itens relevantes por usuário no teste.

    Args:
        test_df: DataFrame com colunas ``user_id``, ``item_id``, ``label``.

    Returns:
        Dicionário user_id → {item_ids relevantes}.
    """
    truth: dict[int, set[int]] = {}
    for uid, iid in zip(test_df["user_id"], test_df["item_id"], strict=True):
        truth.setdefault(uid, set()).add(iid)
    return truth


def _load_neural_model(meta: dict[str, int], models_dir: Path) -> RecommenderModel:
    """Reconstrói o NCF treinado a partir do checkpoint.

    Args:
        meta: Metadados com ``n_users`` e ``n_items``.
        models_dir: Diretório com ``ncf_model.pt``.

    Returns:
        NCF com os pesos treinados carregados.
    """
    ncf = ModelFactory.create(
        "ncf",
        n_users=meta["n_users"],
        n_items=meta["n_items"],
    )
    ncf.load_state_dict(torch.load(models_dir / "ncf_model.pt", weights_only=True))
    return ncf


def _build_models(
    meta: dict[str, int], models_dir: Path
) -> dict[str, RecommenderModel]:
    """Instancia o modelo neural e todos os baselines via Factory.

    Args:
        meta: Metadados do preprocessamento.
        models_dir: Diretório com artefatos de modelos.

    Returns:
        Dicionário nome → modelo pronto para ``fit``/``recommend``.
    """
    models: dict[str, RecommenderModel] = {"ncf": _load_neural_model(meta, models_dir)}
    for name in _BASELINE_NAMES:
        models[name] = ModelFactory.create(name)
    return models


def _evaluate_model(
    model_name: str,
    ground_truth: dict[int, set[int]],
    predictions: dict[int, list[int]],
    k: int,
) -> dict[str, float]:
    """Calcula as 4 métricas de ranking para um modelo.

    Args:
        model_name: Nome do modelo (para log).
        ground_truth: Itens relevantes por usuário.
        predictions: Recomendações geradas por usuário.
        k: Tamanho do ranking.

    Returns:
        Dicionário com as métricas calculadas.
    """
    users = sorted(ground_truth.keys())
    actuals = [ground_truth[u] for u in users]
    preds = [predictions.get(u, []) for u in users]

    p_at_k = sum(
        precision_at_k(a, p, k) for a, p in zip(actuals, preds, strict=True)
    ) / len(users)
    r_at_k = sum(
        recall_at_k(a, p, k) for a, p in zip(actuals, preds, strict=True)
    ) / len(users)
    m_at_k = map_at_k(actuals, preds, k)
    n_at_k = sum(ndcg_at_k(a, p, k) for a, p in zip(actuals, preds, strict=True)) / len(
        users
    )

    metrics = {
        f"{model_name}/precision@{k}": p_at_k,
        f"{model_name}/recall@{k}": r_at_k,
        f"{model_name}/map@{k}": m_at_k,
        f"{model_name}/ndcg@{k}": n_at_k,
    }
    return metrics


def _print_report(all_metrics: dict[str, float], model_names: list[str]) -> None:
    """Imprime a tabela comparativa de métricas por modelo.

    Args:
        all_metrics: Métricas no formato ``modelo/métrica@k``.
        model_names: Modelos avaliados, na ordem das colunas.
    """
    print("📊 Resultados da avaliação:")
    header = f"{'Métrica':<16}" + "".join(f"{name:>14}" for name in model_names)
    print(header)
    print("-" * len(header))
    for metric_key in ("precision", "recall", "map", "ndcg"):
        row = f"  {metric_key}@{_K:<11}"
        for name in model_names:
            row += f"{all_metrics[f'{name}/{metric_key}@{_K}']:>14.4f}"
        print(row)


def run(
    processed_dir: Path | None = None,
    models_dir: Path | None = None,
) -> None:
    """Executa a avaliação comparativa NCF vs baselines.

    Args:
        processed_dir: Diretório com dados processados e metadados.
        models_dir: Diretório com o modelo treinado.
    """
    processed_dir = processed_dir or settings.data_processed_dir
    models_dir = models_dir or settings.models_dir

    # ── Carregar dados ──────────────────────────────────────────
    test_df = pd.read_csv(processed_dir / "test.csv")
    train_df = pd.read_csv(processed_dir / "train.csv")
    with (processed_dir / "metadata.json").open(encoding="utf-8") as f:
        meta = json.load(f)

    ground_truth = _build_ground_truth(test_df)
    train_users = train_df["user_id"].tolist()
    train_items = train_df["item_id"].tolist()

    # ── Treinar/avaliar cada modelo genericamente ───────────────
    models = _build_models(meta, models_dir)
    all_metrics: dict[str, float] = {}
    for name, model in models.items():
        model.fit(user_ids=train_users, item_ids=train_items)
        preds = {uid: model.recommend(uid, k=_K) for uid in ground_truth}
        all_metrics.update(_evaluate_model(name, ground_truth, preds, _K))

    # ── MLflow logging ──────────────────────────────────────────
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    with mlflow.start_run(run_name="evaluation"):
        mlflow.log_metrics({k.replace("@", "_at_"): v for k, v in all_metrics.items()})

    # ── Salvar relatório ────────────────────────────────────────
    report_path = models_dir / "evaluation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    _print_report(all_metrics, list(models))
    print(f"\n✅ Relatório salvo em {report_path}")


if __name__ == "__main__":
    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    sys.exit(run())  # type: ignore[func-returns-value]
