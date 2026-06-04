"""Pré-processadores plugáveis via padrão Strategy."""

from recsys.preprocessing.base import PreprocessingStrategy
from recsys.preprocessing.pipeline import PreprocessingPipeline
from recsys.preprocessing.strategies import (
    IdentityStrategy,
    LabelEncodeStrategy,
)

__all__ = [
    "IdentityStrategy",
    "LabelEncodeStrategy",
    "PreprocessingPipeline",
    "PreprocessingStrategy",
]
