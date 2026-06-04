import pytest

from recsys.models import ModelFactory, PopularityRecommender


def test_cria_modelo_registrado():
    model = ModelFactory.create("popularity")
    assert isinstance(model, PopularityRecommender)


def test_lista_modelos_disponiveis():
    assert "popularity" in ModelFactory.available()


def test_modelo_desconhecido_levanta_value_error():
    with pytest.raises(ValueError, match="não registrado"):
        ModelFactory.create("inexistente")


def test_baseline_recomenda_itens_mais_populares():
    model = ModelFactory.create("popularity")
    model.fit(user_ids=[1, 1, 2], item_ids=[10, 10, 20])
    assert model.recommend(user_id=1, k=1) == [10]
