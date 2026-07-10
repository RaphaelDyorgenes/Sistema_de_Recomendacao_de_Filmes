"""Registro e promoção do modelo no MLflow Model Registry.

Lê o melhor run apontado por ``models/best_run.json`` (gerado pelos
experimentos), registra o modelo como nova versão no Registry e o
promove pelos estágios Staging → Production.

Uso:
    python -m recsys.training.registry
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mlflow
from mlflow import MlflowClient

from recsys.utils.config import settings

_MODEL_NAME = "ncf-recommender"


def _load_best_run(models_dir: Path) -> dict[str, str]:
    """Lê o resultado do melhor experimento.

    Args:
        models_dir: Diretório com ``best_run.json``.

    Returns:
        Dicionário com ``run_id``, ``run_name`` e ``best_val_loss``.

    Raises:
        FileNotFoundError: Se os experimentos ainda não foram executados.
    """
    best_path = models_dir / "best_run.json"
    if not best_path.exists():
        msg = (
            f"{best_path} não encontrado. "
            "Execute antes: python -m recsys.training.experiments"
        )
        raise FileNotFoundError(msg)
    with best_path.open(encoding="utf-8") as f:
        return json.load(f)


def register_model(run_id: str, model_name: str = _MODEL_NAME) -> str:
    """Registra o modelo de um run como nova versão no Registry.

    Args:
        run_id: Run do MLflow que logou o modelo em ``model``.
        model_name: Nome do modelo registrado.

    Returns:
        Número da versão criada.
    """
    version = mlflow.register_model(f"runs:/{run_id}/model", model_name)
    print(f"📦 Registrado '{model_name}' versão {version.version}")
    return version.version


def promote(model_name: str, version: str, stage: str) -> None:
    """Promove uma versão registrada para o estágio informado.

    Args:
        model_name: Nome do modelo no Registry.
        version: Versão a promover.
        stage: Estágio de destino (``Staging`` ou ``Production``).
    """
    client = MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage,
        archive_existing_versions=(stage == "Production"),
    )
    print(f"🚀 '{model_name}' v{version} promovido para {stage}")


def run(models_dir: Path | None = None, model_name: str = _MODEL_NAME) -> None:
    """Registra o melhor modelo e o promove até Production.

    Args:
        models_dir: Diretório com ``best_run.json``.
        model_name: Nome do modelo no Registry.
    """
    models_dir = models_dir or settings.models_dir
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    best = _load_best_run(models_dir)
    print(
        f"🏆 Melhor run: {best['run_name']} "
        f"(val_loss={best['best_val_loss']:.4f}, id={best['run_id']})"
    )

    version = register_model(best["run_id"], model_name)
    promote(model_name, version, "Staging")
    promote(model_name, version, "Production")


if __name__ == "__main__":
    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    run()
