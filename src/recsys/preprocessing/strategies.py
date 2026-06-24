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
    ``fit`` (usando Scikit-Learn) e reaplicado de forma determinística no ``transform``.
    """

    _UNKNOWN = -1

    def __init__(self) -> None:
        """Inicializa o encoder do Scikit-Learn."""
        from sklearn.preprocessing import LabelEncoder

        self._encoder = LabelEncoder()
        self._fitted = False

    def fit(self, values: Sequence[int]) -> None:
        """Constrói o mapa id-bruto -> índice contíguo.

        Args:
            values: Valores brutos de treino.
        """
        self._encoder.fit(values)
        self._fitted = True

    def transform(self, values: Sequence[int]) -> list[int]:
        """Substitui cada id pelo seu índice; desconhecidos viram ``-1``.

        Args:
            values: Valores a codificar.

        Returns:
            Índices contíguos correspondentes.
        """
        if not self._fitted:
            return [self._UNKNOWN] * len(values)

        known_classes = set(self._encoder.classes_)
        # Substituímos os OOV por uma classe conhecida temporariamente
        # para o transform não falhar
        safe_values = [
            v if v in known_classes else self._encoder.classes_[0] for v in values
        ]

        encoded = self._encoder.transform(safe_values)

        return [
            int(e) if orig in known_classes else self._UNKNOWN
            for e, orig in zip(encoded, values, strict=True)
        ]

    @property
    def vocabulary_size(self) -> int:
        """Número de ids distintos aprendidos.

        Returns:
            Tamanho do vocabulário, útil para dimensionar embeddings.
        """
        if not self._fitted:
            return 0
        return len(self._encoder.classes_)
