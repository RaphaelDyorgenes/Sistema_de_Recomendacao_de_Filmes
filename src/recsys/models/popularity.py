"""Baseline de recomendação por popularidade."""

from collections import Counter
from collections.abc import Sequence

from recsys.models.base import RecommenderModel
from recsys.models.factory import ModelFactory


@ModelFactory.register("popularity")
class PopularityRecommender(RecommenderModel):
    """Recomenda os itens mais frequentes do histórico, ignorando o usuário.

    Funciona como piso não-personalizado: qualquer modelo personalizado
    precisa, no mínimo, superá-lo.
    """

    def __init__(self) -> None:
        """Inicializa o ranking vazio (preenchido em ``fit``)."""
        self._ranking: list[int] = []

    def fit(self, user_ids: Sequence[int], item_ids: Sequence[int]) -> None:
        """Calcula o ranking global de itens por frequência.

        Args:
            user_ids: Não usado; o ranking é global, não por usuário.
            item_ids: Itens observados; a frequência define o ranking.
        """
        counts = Counter(item_ids)
        self._ranking = [item for item, _ in counts.most_common()]

    def recommend(self, user_id: int, k: int = 10) -> list[int]:
        """Retorna os ``k`` itens mais populares.

        Args:
            user_id: Não usado; o ranking é o mesmo para todos.
            k: Quantidade de itens a retornar.

        Returns:
            Os ``k`` itens mais frequentes, em ordem decrescente.
        """
        return self._ranking[:k]
