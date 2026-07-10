"""Script de treinamento do modelo NCF — stage ``train`` do DVC.

Orquestra a criação do modelo, carregamento de dados processados,
treinamento via ``Trainer`` e logging no MLflow. Os hiperparâmetros
ficam em ``TrainConfig``, permitindo reuso pelo script de experimentos.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import mlflow
import numpy as np
import torch
from torch.utils.data import DataLoader

from recsys.data.dataset import InteractionDataset
from recsys.models.base import RecommenderModel
from recsys.models.factory import ModelFactory
from recsys.models.ncf import NCFRecommender as _  # noqa: F401 — registra no Factory
from recsys.training.trainer import Trainer
from recsys.utils.config import settings


@dataclass(frozen=True)
class TrainConfig:
    """Hiperparâmetros de um treino do NCF.

    Attributes:
        embedding_dim: Dimensão dos embeddings de usuário e item.
        hidden_layers: Tamanhos das camadas ocultas do MLP.
        dropout: Taxa de dropout entre camadas.
        lr: Taxa de aprendizado do Adam.
        batch_size: Tamanho do batch nos DataLoaders.
        epochs: Número máximo de épocas.
        patience: Épocas sem melhora antes do early stopping.
        run_name: Nome do run no MLflow.
    """

    embedding_dim: int = 32
    hidden_layers: tuple[int, ...] = (64, 32, 16)
    dropout: float = 0.2
    lr: float = 1e-3
    batch_size: int = 512
    epochs: int = 20
    patience: int = 5
    run_name: str = "ncf-train"


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


def _load_metadata(processed_dir: Path) -> dict[str, int]:
    """Lê os metadados gerados pelo preprocessamento.

    Args:
        processed_dir: Diretório com ``metadata.json``.

    Returns:
        Metadados com ``n_users`` e ``n_items``.
    """
    with (processed_dir / "metadata.json").open(encoding="utf-8") as f:
        return json.load(f)


def _build_loaders(
    processed_dir: Path, batch_size: int
) -> tuple[DataLoader, DataLoader]:
    """Monta os DataLoaders de treino e validação.

    Args:
        processed_dir: Diretório com os CSVs processados.
        batch_size: Tamanho do batch.

    Returns:
        Tupla ``(train_loader, val_loader)``.
    """
    train_ds = InteractionDataset(processed_dir / "train_with_negatives.csv")
    val_ds = InteractionDataset(processed_dir / "val.csv")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader


def _create_model(meta: dict[str, int], config: TrainConfig) -> RecommenderModel:
    """Instancia o NCF via Factory com os hiperparâmetros da config.

    Args:
        meta: Metadados com os tamanhos de vocabulário.
        config: Hiperparâmetros do treino.

    Returns:
        Modelo NCF pronto para treinar.
    """
    return ModelFactory.create(
        "ncf",
        n_users=meta["n_users"],
        n_items=meta["n_items"],
        embedding_dim=config.embedding_dim,
        hidden_layers=config.hidden_layers,
        dropout=config.dropout,
    )


def run(
    config: TrainConfig | None = None,
    processed_dir: Path | None = None,
    models_dir: Path | None = None,
    seed: int | None = None,
) -> tuple[str, float]:
    """Executa um treinamento completo do NCF como um run MLflow.

    Args:
        config: Hiperparâmetros do treino. Padrão: ``TrainConfig()``.
        processed_dir: Diretório com dados processados.
        models_dir: Diretório para salvar o modelo treinado.
        seed: Semente para reprodutibilidade.

    Returns:
        Tupla ``(run_id, best_val_loss)`` do run executado.
    """
    config = config or TrainConfig()
    processed_dir = processed_dir or settings.data_processed_dir
    models_dir = models_dir or settings.models_dir
    seed = seed if seed is not None else settings.random_seed

    _set_seeds(seed)
    meta = _load_metadata(processed_dir)
    train_loader, val_loader = _build_loaders(processed_dir, config.batch_size)
    model = _create_model(meta, config)

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    with mlflow.start_run(run_name=config.run_name) as active_run:
        best_val_loss = _train_and_save(
            model, train_loader, val_loader, config, models_dir, processed_dir
        )
        print(f"   MLflow run: {active_run.info.run_id}")
        return active_run.info.run_id, best_val_loss


def _train_and_save(
    model: RecommenderModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainConfig,
    models_dir: Path,
    processed_dir: Path,
) -> float:
    """Treina o modelo, salva o checkpoint final e loga os artefatos.

    Args:
        model: Modelo NCF a treinar.
        train_loader: DataLoader de treino.
        val_loader: DataLoader de validação.
        config: Hiperparâmetros do treino.
        models_dir: Diretório de saída do checkpoint.
        processed_dir: Diretório com ``metadata.json``.

    Returns:
        Melhor loss de validação alcançada.
    """
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=config.lr,
        epochs=config.epochs,
        patience=config.patience,
        checkpoint_dir=models_dir,
    )
    trainer.fit()

    final_path = models_dir / "ncf_model.pt"
    torch.save(model.state_dict(), final_path)

    mlflow.log_artifact(str(final_path))
    mlflow.log_artifact(str(processed_dir / "metadata.json"))

    # Modelo no formato MLflow: habilita o registro de versões no Registry.
    # Serialização pickle: o formato pt2 exige assinatura via TensorSpec
    # e não suporta o forward com dois tensores (user_ids, item_ids).
    mlflow.pytorch.log_model(model, name="model", serialization_format="pickle")

    print(f"✅ Modelo salvo em {final_path}")
    return trainer.best_val_loss


if __name__ == "__main__":
    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    run()
