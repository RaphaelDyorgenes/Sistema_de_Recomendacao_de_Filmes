"""Interface das estratégias de pré-processamento."""

from abc import ABC, abstractmethod
from collections.abc import Sequence


class PreprocessingStrategy(ABC):
    """Contrato de uma transformação aplicada a uma coluna de ids.

    Cada estratégia recebe uma sequência de valores brutos e devolve a
    versão transformada, sem conhecer as demais nem a ordem em que rodam.
    """

    @abstractmethod
    def fit(self, values: Sequence[int]) -> None:
        """Aprende o estado necessário a partir dos dados de treino.

        Args:
            values: Valores brutos observados no treino.
        """

    @abstractmethod
    def transform(self, values: Sequence[int]) -> list[int]:
        """Aplica a transformação aprendida.

        Args:
            values: Valores a transformar.

        Returns:
            Valores transformados, na mesma ordem da entrada.
        """
