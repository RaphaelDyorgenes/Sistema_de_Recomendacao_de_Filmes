"""Interface base para modelos de recomendação."""

from abc import ABC, abstractmethod
from collections.abc import Sequence


class RecommenderModel(ABC):
    """Contrato comum a todo modelo de recomendação.

    Define a fronteira estável que treino, avaliação e serialização enxergam,
    sem se acoplar a se o modelo é uma rede neural ou um baseline clássico.
    """

    @abstractmethod
    def fit(self, user_ids: Sequence[int], item_ids: Sequence[int]) -> None:
        """Treina o modelo a partir de interações usuário-item.

        Args:
            user_ids: Ids de usuário de cada interação observada.
            item_ids: Ids de item correspondentes, alinhados a ``user_ids``.
        """

    @abstractmethod
    def recommend(self, user_id: int, k: int = 10) -> list[int]:
        """Recomenda os ``k`` itens mais relevantes para um usuário.

        Args:
            user_id: Usuário-alvo da recomendação.
            k: Quantidade de itens a retornar.

        Returns:
            Ids de item ordenados por relevância decrescente.
        """
