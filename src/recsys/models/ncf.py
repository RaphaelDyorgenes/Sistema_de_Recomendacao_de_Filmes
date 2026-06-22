"""Modelo Neural Collaborative Filtering (NCF) para recomendação de produtos.

Combina embeddings de usuário e item em um MLP multicamadas para prever
a probabilidade de interação (clique / navegação) em um e-commerce.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn

from recsys.models.base import RecommenderModel
from recsys.models.factory import ModelFactory


@ModelFactory.register("ncf")
class NCFRecommender(RecommenderModel, nn.Module):
    """Neural Collaborative Filtering baseado em embeddings + MLP.

    Arquitetura:
        1. Embedding de usuário (dim ``embedding_dim``)
        2. Embedding de item   (dim ``embedding_dim``)
        3. Concatenação → MLP com camadas decrescentes → sigmoid

    O modelo aprende representações latentes que capturam o padrão de
    navegação dos usuários e a similaridade entre produtos.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 32,
        hidden_layers: Sequence[int] = (64, 32, 16),
        dropout: float = 0.2,
    ) -> None:
        """Constrói as camadas do NCF.

        Args:
            n_users: Número de usuários distintos (tamanho do vocabulário).
            n_items: Número de itens distintos (tamanho do vocabulário).
            embedding_dim: Dimensão dos vetores de embedding.
            hidden_layers: Tamanhos das camadas ocultas do MLP.
            dropout: Taxa de dropout entre camadas.
        """
        nn.Module.__init__(self)

        self._n_users = n_users
        self._n_items = n_items
        self._embedding_dim = embedding_dim

        # ── Embeddings ──────────────────────────────────────────
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        # ── MLP ─────────────────────────────────────────────────
        layers: list[nn.Module] = []
        input_dim = embedding_dim * 2
        for hidden_dim in hidden_layers:
            layers.extend(
                [
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))

        self.mlp = nn.Sequential(*layers)

        # ── Inicialização Xavier ────────────────────────────────
        self._init_weights()

        # ── Estado para recommend() ─────────────────────────────
        self._all_item_ids: list[int] = []
        self._user_history: dict[int, set[int]] = {}

    def _init_weights(self) -> None:
        """Aplica inicialização Xavier Uniform aos pesos lineares."""
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """Calcula a probabilidade de interação para pares (user, item).

        Args:
            user_ids: Tensor de ids de usuário ``(batch_size,)``.
            item_ids: Tensor de ids de item ``(batch_size,)``.

        Returns:
            Tensor ``(batch_size,)`` com probabilidades em ``[0, 1]``.
        """
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        x = torch.cat([user_emb, item_emb], dim=-1)
        logits = self.mlp(x).squeeze(-1)
        return torch.sigmoid(logits)

    def fit(self, user_ids: Sequence[int], item_ids: Sequence[int]) -> None:
        """Registra o histórico de interações para uso em ``recommend``.

        O treinamento real dos pesos acontece via ``Trainer``;
        este método apenas guarda o mapa de interações para filtragem.

        Args:
            user_ids: Ids de usuário de cada interação.
            item_ids: Ids de item correspondentes.
        """
        self._all_item_ids = sorted(set(item_ids))
        self._user_history = {}
        for uid, iid in zip(user_ids, item_ids, strict=True):
            self._user_history.setdefault(uid, set()).add(iid)

    def recommend(self, user_id: int, k: int = 10) -> list[int]:
        """Recomenda os ``k`` itens com maior score para o usuário.

        Exclui itens já vistos pelo usuário para evitar recomendações
        triviais. Usa inferência em batch para eficiência.

        Args:
            user_id: Usuário-alvo.
            k: Número de itens a retornar.

        Returns:
            Ids dos ``k`` itens mais relevantes, em ordem decrescente.
        """
        self.eval()
        seen = self._user_history.get(user_id, set())
        candidates = [i for i in self._all_item_ids if i not in seen]

        if not candidates:
            return []

        with torch.no_grad():
            user_tensor = torch.tensor([user_id] * len(candidates), dtype=torch.long)
            item_tensor = torch.tensor(candidates, dtype=torch.long)
            scores = self.forward(user_tensor, item_tensor).cpu().numpy()

        top_indices = np.argsort(scores)[::-1][:k]
        return [candidates[int(i)] for i in top_indices]

    def get_hparams(self) -> dict[str, int | float | str]:
        """Retorna os hiperparâmetros do modelo para logging no MLflow.

        Returns:
            Dicionário com os hiperparâmetros configurados.
        """
        return {
            "n_users": self._n_users,
            "n_items": self._n_items,
            "embedding_dim": self._embedding_dim,
            "hidden_layers": str(
                [m.out_features for m in self.mlp if isinstance(m, nn.Linear)]
            ),
            "model_type": "ncf",
        }
