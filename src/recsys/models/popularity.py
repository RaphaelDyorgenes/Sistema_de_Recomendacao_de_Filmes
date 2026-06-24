"""Baseline de recomendação por popularidade."""

from collections.abc import Sequence

import numpy as np
from sklearn.dummy import DummyClassifier

from recsys.models.base import RecommenderModel
from recsys.models.factory import ModelFactory


@ModelFactory.register("popularity")
class PopularityRecommender(RecommenderModel):
    """Recomenda os itens mais frequentes do histórico, ignorando o usuário.

    Funciona como piso não-personalizado: qualquer modelo personalizado
    precisa, no mínimo, superá-lo.
    """

    def __init__(self) -> None:
        """Inicializa o modelo DummyClassifier do Scikit-Learn."""
        self._clf = DummyClassifier(strategy="prior")
        self._ranking: list[int] = []

    def fit(self, user_ids: Sequence[int], item_ids: Sequence[int]) -> None:
        """Calcula o ranking global de itens por frequência usando Scikit-Learn.

        Args:
            user_ids: Não usado; o ranking é global, não por usuário.
            item_ids: Itens observados; a frequência define o ranking.
        """
        # Scikit-Learn exige um X de formato (n_samples, n_features)
        x_dummy = np.zeros((len(item_ids), 1))
        self._clf.fit(x_dummy, item_ids)

        # A propriedade class_prior_ contém a frequência de cada item
        # listado em classes_
        priors = self._clf.class_prior_
        classes = self._clf.classes_

        # Ordenamos os índices pela probabilidade (decrescente)
        sorted_indices = np.argsort(priors)[::-1]
        self._ranking = [int(classes[i]) for i in sorted_indices]

    def recommend(self, user_id: int, k: int = 10) -> list[int]:
        """Retorna os ``k`` itens mais populares.

        Args:
            user_id: Não usado; o ranking é o mesmo para todos.
            k: Quantidade de itens a retornar.

        Returns:
            Os ``k`` itens mais frequentes, em ordem decrescente.
        """
        return self._ranking[:k]
