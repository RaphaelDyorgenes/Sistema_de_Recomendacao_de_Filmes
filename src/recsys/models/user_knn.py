"""Baseline colaborativo k-NN de usuários construído com Scikit-Learn.

Recomenda itens consumidos pelos vizinhos mais similares do usuário
(similaridade cosseno sobre a matriz esparsa usuário-item). Serve como
baseline personalizado: mais forte que popularidade pura, porém sem
aprendizado de representações como o NCF.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MultiLabelBinarizer

from recsys.models.base import RecommenderModel
from recsys.models.factory import ModelFactory


@ModelFactory.register("user_knn")
class UserKNNRecommender(RecommenderModel):
    """Filtragem colaborativa user-based com vizinhos mais próximos.

    Funcionamento:
        1. ``MultiLabelBinarizer`` monta a matriz esparsa usuário-item.
        2. ``NearestNeighbors`` (cosseno) encontra usuários similares.
        3. Itens dos vizinhos são pontuados pela similaridade acumulada.

    Usuários fora do histórico (cold start) recebem os itens mais
    populares como fallback.
    """

    def __init__(self, n_neighbors: int = 20) -> None:
        """Configura o número de vizinhos considerados.

        Args:
            n_neighbors: Quantidade de usuários similares consultados.
        """
        self._n_neighbors = n_neighbors
        self._binarizer = MultiLabelBinarizer(sparse_output=True)
        self._knn = NearestNeighbors(metric="cosine")
        self._matrix = None
        self._user_index: dict[int, int] = {}
        self._popularity: list[int] = []

    def fit(self, user_ids: Sequence[int], item_ids: Sequence[int]) -> None:
        """Constrói a matriz usuário-item e indexa os vizinhos.

        Args:
            user_ids: Ids de usuário de cada interação observada.
            item_ids: Ids de item correspondentes, alinhados a ``user_ids``.
        """
        histories: dict[int, set[int]] = {}
        for uid, iid in zip(user_ids, item_ids, strict=True):
            histories.setdefault(uid, set()).add(iid)

        users = sorted(histories)
        self._user_index = {uid: pos for pos, uid in enumerate(users)}
        self._matrix = self._binarizer.fit_transform([histories[u] for u in users])

        # +1 porque o vizinho mais próximo de um usuário é ele mesmo.
        n_lookup = min(self._n_neighbors + 1, len(users))
        self._knn = NearestNeighbors(metric="cosine", n_neighbors=n_lookup)
        self._knn.fit(self._matrix)
        self._popularity = self._rank_by_popularity()

    def recommend(self, user_id: int, k: int = 10) -> list[int]:
        """Recomenda os ``k`` itens com maior score de vizinhança.

        Args:
            user_id: Usuário-alvo da recomendação.
            k: Quantidade de itens a retornar.

        Returns:
            Ids de item em ordem decrescente de relevância. Usuários sem
            histórico recebem os itens mais populares.
        """
        if user_id not in self._user_index:
            return self._popularity[:k]

        row_idx = self._user_index[user_id]
        scores = self._score_candidates(row_idx)

        seen_mask = self._matrix[row_idx].toarray().ravel() > 0
        scores[seen_mask] = -np.inf

        top = np.argsort(scores)[::-1][:k]
        recs = [int(self._binarizer.classes_[i]) for i in top if scores[i] > 0]

        seen_ids = {
            int(self._binarizer.classes_[i]) for i in self._matrix[row_idx].indices
        }
        return self._fill_with_popular(recs, seen_ids, k)

    def _score_candidates(self, row_idx: int) -> np.ndarray:
        """Pontua todos os itens pela similaridade acumulada dos vizinhos.

        Args:
            row_idx: Índice da linha do usuário na matriz usuário-item.

        Returns:
            Vetor denso ``(n_items,)`` com o score de cada item.
        """
        distances, indices = self._knn.kneighbors(self._matrix[row_idx])
        weights = 1.0 - distances.ravel()
        neighbors = indices.ravel()

        keep = neighbors != row_idx
        weights, neighbors = weights[keep], neighbors[keep]

        weighted = self._matrix[neighbors].multiply(weights[:, None])
        return np.asarray(weighted.sum(axis=0)).ravel()

    def _rank_by_popularity(self) -> list[int]:
        """Ordena os itens por frequência global (fallback de cold start).

        Returns:
            Ids de item do mais para o menos popular.
        """
        counts = np.asarray(self._matrix.sum(axis=0)).ravel()
        order = np.argsort(counts)[::-1]
        return [int(self._binarizer.classes_[i]) for i in order]

    def _fill_with_popular(self, recs: list[int], seen: set[int], k: int) -> list[int]:
        """Completa a lista até ``k`` itens usando o ranking de popularidade.

        Args:
            recs: Recomendações já geradas pela vizinhança.
            seen: Itens que o usuário já consumiu (não recomendar).
            k: Tamanho final desejado da lista.

        Returns:
            Lista com até ``k`` itens, sem repetições nem itens vistos.
        """
        for item in self._popularity:
            if len(recs) >= k:
                break
            if item not in seen and item not in recs:
                recs.append(item)
        return recs[:k]

    def get_hparams(self) -> dict[str, int | str]:
        """Retorna os hiperparâmetros do baseline para logging.

        Returns:
            Dicionário com os hiperparâmetros configurados.
        """
        return {"n_neighbors": self._n_neighbors, "model_type": "user_knn"}
