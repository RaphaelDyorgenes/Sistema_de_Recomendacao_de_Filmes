"""Interface base para modelos de recomendação."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any


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

    def save(self, path: Path) -> None:
        """Persiste o modelo em disco.

        A implementação padrão usa ``pickle``; subclasses PyTorch devem
        sobrescrever para usar ``torch.save``.

        Args:
            path: Caminho do arquivo de saída.
        """
        import pickle

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "RecommenderModel":
        """Carrega um modelo previamente salvo com ``save``.

        Args:
            path: Caminho do arquivo serializado.

        Returns:
            Instância do modelo reconstruída.
        """
        import pickle

        with path.open("rb") as f:
            return pickle.load(f)  # noqa: S301

    def get_hparams(self) -> dict[str, Any]:
        """Retorna hiperparâmetros para logging (MLflow, etc.).

        Returns:
            Dicionário vazio por padrão; subclasses sobrescrevem.
        """
        return {}
