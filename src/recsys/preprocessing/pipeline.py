"""Contexto que orquestra estratégias de pré-processamento."""

from collections.abc import Sequence

from recsys.preprocessing.base import PreprocessingStrategy


class PreprocessingPipeline:
    """Aplica uma sequência de estratégias em ordem.

    É o contexto do padrão Strategy: depende só da interface
    ``PreprocessingStrategy``, então novas transformações entram sem que
    esta classe mude.
    """

    def __init__(self, strategies: Sequence[PreprocessingStrategy]) -> None:
        """Recebe as estratégias já configuradas, na ordem de aplicação.

        Args:
            strategies: Estratégias a aplicar, da primeira à última.
        """
        self._strategies = list(strategies)

    def fit(self, values: Sequence[int]) -> None:
        """Ajusta cada estratégia sobre a saída da anterior.

        Args:
            values: Valores brutos de treino.
        """
        current = list(values)
        for strategy in self._strategies:
            strategy.fit(current)
            current = strategy.transform(current)

    def transform(self, values: Sequence[int]) -> list[int]:
        """Encadeia ``transform`` de todas as estratégias.

        Args:
            values: Valores a transformar.

        Returns:
            Resultado após a última estratégia.
        """
        current = list(values)
        for strategy in self._strategies:
            current = strategy.transform(current)
        return current
