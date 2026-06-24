"""Avaliação comparativa do modelo NCF contra o baseline de popularidade.

Stage ``evaluate`` do DVC. Gera recomendações para cada usuário do
conjunto de teste, calcula 4 métricas de ranking e loga os resultados
no MLflow para comparação direta.
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
from recsys.models.factory import ModelFactory
from recsys.models.ncf import NCFRecommender as _  # noqa: F401 — registra no Factory
from recsys.models.popularity import PopularityRecommender as __  # noqa: F401
from recsys.utils.config import settings

_K = 10


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


def run(
    processed_dir: Path | None = None,
    models_dir: Path | None = None,
) -> None:
    """Executa a avaliação comparativa NCF vs Popularity.

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

    # ── NCF: carregar e gerar recomendações ─────────────────────
    ncf = ModelFactory.create(
        "ncf",
        n_users=meta["n_users"],
        n_items=meta["n_items"],
    )
    ncf.load_state_dict(torch.load(models_dir / "ncf_model.pt", weights_only=True))
    ncf.fit(
        user_ids=train_df["user_id"].tolist(),
        item_ids=train_df["item_id"].tolist(),
    )

    ncf_preds: dict[int, list[int]] = {}
    for uid in ground_truth:
        ncf_preds[uid] = ncf.recommend(uid, k=_K)

    # ── Popularity: treinar e gerar recomendações ───────────────
    pop = ModelFactory.create("popularity")
    pop.fit(
        user_ids=train_df["user_id"].tolist(),
        item_ids=train_df["item_id"].tolist(),
    )

    pop_preds: dict[int, list[int]] = {}
    for uid in ground_truth:
        pop_preds[uid] = pop.recommend(uid, k=_K)

    # ── Calcular métricas ───────────────────────────────────────
    ncf_metrics = _evaluate_model("ncf", ground_truth, ncf_preds, _K)
    pop_metrics = _evaluate_model("popularity", ground_truth, pop_preds, _K)

    all_metrics = {**ncf_metrics, **pop_metrics}

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

    print("📊 Resultados da avaliação:")
    print(f"{'Métrica':<30} {'NCF':>10} {'Popularity':>12}")
    print("-" * 55)
    for metric_key in ["precision", "recall", "map", "ndcg"]:
        ncf_val = all_metrics.get(f"ncf/{metric_key}@{_K}", 0.0)
        pop_val = all_metrics.get(f"popularity/{metric_key}@{_K}", 0.0)
        print(f"  {metric_key}@{_K:<22} {ncf_val:>10.4f} {pop_val:>12.4f}")

    print(f"\n✅ Relatório salvo em {report_path}")


if __name__ == "__main__":
    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    sys.exit(run())  # type: ignore[func-returns-value]
