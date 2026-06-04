"""Modelos de recomendação e fábrica de criação."""

from recsys.models.base import RecommenderModel
from recsys.models.factory import ModelFactory
from recsys.models.popularity import PopularityRecommender

__all__ = ["ModelFactory", "PopularityRecommender", "RecommenderModel"]
