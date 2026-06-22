"""Dataset PyTorch para interações de recomendação usuário-item."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset


class InteractionDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Carrega pares (user_id, item_id, label) de um CSV processado.

    Cada amostra retorna três tensores escalares que alimentam o
    ``DataLoader`` e, por consequência, o ``forward`` do NCF.
    """

    def __init__(self, csv_path: Path) -> None:
        """Lê o CSV e armazena as colunas como tensores.

        Args:
            csv_path: Caminho para o CSV com colunas
                      ``user_id``, ``item_id``, ``label``.
        """
        df = pd.read_csv(csv_path)
        self._users = torch.tensor(df["user_id"].values, dtype=torch.long)
        self._items = torch.tensor(df["item_id"].values, dtype=torch.long)
        self._labels = torch.tensor(df["label"].values, dtype=torch.float32)

    def __len__(self) -> int:
        """Número total de interações no dataset."""
        return len(self._users)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retorna a interação no índice ``idx``.

        Args:
            idx: Índice da amostra.

        Returns:
            Tupla ``(user_id, item_id, label)``.
        """
        return self._users[idx], self._items[idx], self._labels[idx]
