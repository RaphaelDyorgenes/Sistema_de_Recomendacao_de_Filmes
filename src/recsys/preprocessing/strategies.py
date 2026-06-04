"""Estratégias concretas de pré-processamento."""

from collections.abc import Sequence

from recsys.preprocessing.base import PreprocessingStrategy


class IdentityStrategy(PreprocessingStrategy):
    """Passa os valores adiante sem alterá-los.

    Útil como neutro em testes e quando uma coluna já vem pronta, evitando
    ramos condicionais no código que consome.
    """

    def fit(self, values: Sequence[int]) -> None:
        """Não aprende nada; existe para cumprir o contrato.

        Args:
            values: Ignorado.
        """

    def transform(self, values: Sequence[int]) -> list[int]:
        """Devolve uma cópia dos valores de entrada.

        Args:
            values: Valores a repassar.

        Returns:
            Cópia da entrada como lista.
        """
        return list(values)


class LabelEncodeStrategy(PreprocessingStrategy):
    """Mapeia ids esparsos para um intervalo contíguo ``0..n-1``.

    Redes neurais indexam embeddings por posição, então ids como
    ``[10, 5000, 73]`` precisam virar ``[0, 1, 2]``. O mapa é aprendido no
    ``fit`` e reaplicado de forma determinística no ``transform``.
    """

    _UNKNOWN = -1

    def __init__(self) -> None:
        """Inicializa o mapa vazio (preenchido em ``fit``)."""
        self._mapping: dict[int, int] = {}

    def fit(self, values: Sequence[int]) -> None:
        """Constrói o mapa id-bruto -> índice contíguo.

        A ordenação dos ids únicos torna o mapa estável entre execuções.

        Args:
            values: Valores brutos de treino.
        """
        unique_sorted = sorted(set(values))
        self._mapping = {raw: idx for idx, raw in enumerate(unique_sorted)}

    def transform(self, values: Sequence[int]) -> list[int]:
        """Substitui cada id pelo seu índice; desconhecidos viram ``-1``.

        Args:
            values: Valores a codificar.

        Returns:
            Índices contíguos correspondentes.
        """
        return [self._mapping.get(value, self._UNKNOWN) for value in values]

    @property
    def vocabulary_size(self) -> int:
        """Número de ids distintos aprendidos.

        Returns:
            Tamanho do vocabulário, útil para dimensionar embeddings.
        """
        return len(self._mapping)
