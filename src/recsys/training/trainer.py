"""Loop de treinamento com Early Stopping e integração MLflow.

O ``Trainer`` é agnóstico ao modelo concreto — recebe qualquer
``nn.Module`` com a mesma interface ``forward(user_ids, item_ids)``
e cuida do loop de treino, validação, checkpointing e logging.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import mlflow
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from recsys.utils.config import settings

if TYPE_CHECKING:
    from recsys.models.ncf import NCFRecommender


class EarlyStopping:
    """Monitora a loss de validação e interrompe quando estagna.

    Attributes:
        patience: Épocas sem melhora antes de parar.
        min_delta: Melhora mínima para ser considerada significativa.
    """

    def __init__(self, patience: int = 5, min_delta: float = 1e-4) -> None:
        """Inicializa o monitor.

        Args:
            patience: Épocas toleradas sem melhora.
            min_delta: Diferença mínima para considerar como melhora.
        """
        self.patience = patience
        self.min_delta = min_delta
        self._best_loss: float = float("inf")
        self._counter = 0

    def should_stop(self, val_loss: float) -> bool:
        """Verifica se o treino deve ser interrompido.

        Args:
            val_loss: Loss de validação da época atual.

        Returns:
            ``True`` se o treino deve parar.
        """
        if val_loss < self._best_loss - self.min_delta:
            self._best_loss = val_loss
            self._counter = 0
            return False
        self._counter += 1
        return self._counter >= self.patience


class Trainer:
    """Orquestra o treinamento de um modelo NCF.

    Responsabilidades:
    - Loop de treino/validação
    - Early stopping
    - Logging de métricas e parâmetros no MLflow
    - Checkpointing do melhor modelo
    """

    def __init__(
        self,
        model: NCFRecommender,
        train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        lr: float = 1e-3,
        epochs: int = 20,
        patience: int = 5,
        checkpoint_dir: Path | None = None,
    ) -> None:
        """Configura o trainer.

        Args:
            model: Modelo NCF a treinar.
            train_loader: DataLoader do conjunto de treino.
            val_loader: DataLoader do conjunto de validação.
            lr: Taxa de aprendizado.
            epochs: Número máximo de épocas.
            patience: Épocas para early stopping.
            checkpoint_dir: Diretório para salvar checkpoints.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.lr = lr

        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.early_stopping = EarlyStopping(patience=patience)
        self.checkpoint_dir = checkpoint_dir or settings.models_dir
        self._best_val_loss = float("inf")

    @property
    def best_val_loss(self) -> float:
        """Melhor loss de validação observada no último ``fit``."""
        return self._best_val_loss

    def _train_epoch(self) -> float:
        """Executa uma época de treino.

        Returns:
            Loss média da época.
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for users, items, labels in self.train_loader:
            self.optimizer.zero_grad()
            predictions = self.model(users, items)
            loss = self.criterion(predictions, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def _validate(self) -> float:
        """Executa uma época de validação.

        Returns:
            Loss média de validação.
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for users, items, labels in self.val_loader:
                predictions = self.model(users, items)
                loss = self.criterion(predictions, labels)
                total_loss += loss.item()
                n_batches += 1

        return total_loss / max(n_batches, 1)

    def fit(self) -> Path:
        """Executa o loop completo de treino com logging MLflow.

        Returns:
            Caminho do checkpoint do melhor modelo.
        """
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        best_path = self.checkpoint_dir / "ncf_best.pt"
        self._best_val_loss = float("inf")

        # ── Log de hiperparâmetros ──────────────────────────────
        mlflow.log_params(
            {
                **self.model.get_hparams(),
                "lr": self.lr,
                "epochs_max": self.epochs,
                "patience": self.early_stopping.patience,
                "batch_size": self.train_loader.batch_size,
            }
        )

        for epoch in range(1, self.epochs + 1):
            train_loss = self._train_epoch()
            val_loss = self._validate()

            # Log no MLflow
            mlflow.log_metrics(
                {"train_loss": train_loss, "val_loss": val_loss}, step=epoch
            )
            print(
                f"  Época {epoch:3d}/{self.epochs} | "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f}"
            )

            # Checkpoint do melhor modelo
            if val_loss < self._best_val_loss:
                self._best_val_loss = val_loss
                torch.save(self.model.state_dict(), best_path)

            # Early stopping
            if self.early_stopping.should_stop(val_loss):
                print(f"  ⏹️  Early stopping na época {epoch}")
                break

        # Carregar melhor checkpoint
        self.model.load_state_dict(torch.load(best_path, weights_only=True))
        mlflow.log_metric("best_val_loss", self._best_val_loss)
        print(f"✅ Treino concluído! Melhor val_loss: {self._best_val_loss:.4f}")

        return best_path
