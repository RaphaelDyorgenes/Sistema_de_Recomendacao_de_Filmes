"""Script de treinamento do modelo NCF — stage ``train`` do DVC.

Orquestra a criação do modelo, carregamento de dados processados,
treinamento via ``Trainer`` e logging no MLflow.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mlflow
import numpy as np
import torch
from torch.utils.data import DataLoader

from recsys.data.dataset import InteractionDataset
from recsys.models.factory import ModelFactory
from recsys.models.ncf import NCFRecommender as _  # noqa: F401 — registra no Factory
from recsys.training.trainer import Trainer
from recsys.utils.config import settings

# ── Hiperparâmetros padrão ──────────────────────────────────────
_BATCH_SIZE = 512
_LR = 1e-3
_EPOCHS = 20
_PATIENCE = 5
_EMBEDDING_DIM = 32
_HIDDEN_LAYERS = (64, 32, 16)
_DROPOUT = 0.2


def _set_seeds(seed: int) -> None:
    """Fixa sementes para reprodutibilidade total.

    Args:
        seed: Valor da semente.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def run(
    processed_dir: Path | None = None,
    models_dir: Path | None = None,
    seed: int | None = None,
) -> None:
    """Executa o treinamento completo do NCF.

    Args:
        processed_dir: Diretório com dados processados.
        models_dir: Diretório para salvar o modelo treinado.
        seed: Semente para reprodutibilidade.
    """
    processed_dir = processed_dir or settings.data_processed_dir
    models_dir = models_dir or settings.models_dir
    seed = seed if seed is not None else settings.random_seed

    _set_seeds(seed)

    # ── Carregar metadados ──────────────────────────────────────
    with (processed_dir / "metadata.json").open(encoding="utf-8") as f:
        meta = json.load(f)

    n_users = meta["n_users"]
    n_items = meta["n_items"]

    # ── Datasets e DataLoaders ──────────────────────────────────
    train_ds = InteractionDataset(processed_dir / "train_with_negatives.csv")
    val_ds = InteractionDataset(processed_dir / "val.csv")

    train_loader = DataLoader(
        train_ds, batch_size=_BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=_BATCH_SIZE, shuffle=False, num_workers=0
    )

    # ── Criar modelo via Factory ────────────────────────────────
    model = ModelFactory.create(
        "ncf",
        n_users=n_users,
        n_items=n_items,
        embedding_dim=_EMBEDDING_DIM,
        hidden_layers=_HIDDEN_LAYERS,
        dropout=_DROPOUT,
    )

    # ── MLflow experiment ───────────────────────────────────────
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    with mlflow.start_run(run_name="ncf-train"):
        # ── Treinar ─────────────────────────────────────────────
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            lr=_LR,
            epochs=_EPOCHS,
            patience=_PATIENCE,
            checkpoint_dir=models_dir,
        )
        trainer.fit()

        # ── Salvar modelo final ─────────────────────────────────
        final_path = models_dir / "ncf_model.pt"
        torch.save(model.state_dict(), final_path)

        # ── Logar artefato no MLflow ────────────────────────────
        mlflow.log_artifact(str(final_path))
        mlflow.log_artifact(str(processed_dir / "metadata.json"))

        print(f"✅ Modelo salvo em {final_path}")
        print(f"   MLflow run: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    sys.exit(run())  # type: ignore[func-returns-value]
