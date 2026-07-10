"""Experimentos de hiperparâmetros do NCF rastreados no MLflow.

Treina o modelo com múltiplas configurações (um run MLflow por config,
com parâmetros, métricas e artefatos) e salva em ``models/best_run.json``
o run com menor loss de validação, usado depois pelo Model Registry.

Uso:
    python -m recsys.training.experiments
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TypedDict

from recsys.training.train import TrainConfig
from recsys.training.train import run as train_run
from recsys.utils.config import settings

# Configurações comparadas: capacidade crescente de representação,
# com learning rate menor para a rede maior.
_CONFIGS: tuple[TrainConfig, ...] = (
    TrainConfig(run_name="ncf-emb16", embedding_dim=16, hidden_layers=(32, 16, 8)),
    TrainConfig(run_name="ncf-emb32", embedding_dim=32, hidden_layers=(64, 32, 16)),
    TrainConfig(
        run_name="ncf-emb64",
        embedding_dim=64,
        hidden_layers=(128, 64, 32),
        lr=5e-4,
    ),
)


class ExperimentResult(TypedDict):
    """Resultado resumido de um run de treinamento."""

    run_name: str
    run_id: str
    best_val_loss: float


def _save_best(best: ExperimentResult, models_dir: Path) -> Path:
    """Persiste o melhor resultado para consumo pelo registro de modelos.

    Args:
        best: Resultado do run com menor loss de validação.
        models_dir: Diretório de artefatos de modelos.

    Returns:
        Caminho do arquivo ``best_run.json`` gerado.
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    best_path = models_dir / "best_run.json"
    with best_path.open("w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)
    return best_path


def _print_summary(results: list[ExperimentResult], best: ExperimentResult) -> None:
    """Imprime a tabela comparativa dos experimentos.

    Args:
        results: Resultados de todos os runs executados.
        best: Run vencedor.
    """
    print("\n📊 Comparativo dos experimentos:")
    print(f"{'Run':<14}{'val_loss':>12}")
    print("-" * 26)
    for result in results:
        marker = " ⭐" if result["run_id"] == best["run_id"] else ""
        print(f"{result['run_name']:<14}{result['best_val_loss']:>12.4f}{marker}")


def run(models_dir: Path | None = None) -> ExperimentResult:
    """Executa todos os experimentos e devolve o melhor.

    Args:
        models_dir: Diretório de artefatos. Padrão: ``settings.models_dir``.

    Returns:
        Resultado do run com menor loss de validação.
    """
    models_dir = models_dir or settings.models_dir

    results: list[ExperimentResult] = []
    for config in _CONFIGS:
        print(f"\n🚀 Experimento: {config.run_name}")
        run_id, val_loss = train_run(config=config, models_dir=models_dir)
        results.append(
            {"run_name": config.run_name, "run_id": run_id, "best_val_loss": val_loss}
        )

    best = min(results, key=lambda r: r["best_val_loss"])
    best_path = _save_best(best, models_dir)
    _print_summary(results, best)
    print(f"\n✅ Melhor run salvo em {best_path}")
    return best


if __name__ == "__main__":
    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    run()
