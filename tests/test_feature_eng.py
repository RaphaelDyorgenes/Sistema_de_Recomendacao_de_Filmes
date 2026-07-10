import numpy as np
import pandas as pd

from recsys.data.feature_eng import _generate_negatives, _positives_by_user
from recsys.data.preprocess import _sample_active_users


def _df_interacoes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": [1, 1, 2],
            "item_id": [0, 1, 2],
            "label": [1, 1, 1],
        }
    )


def test_positives_by_user_agrupa_itens():
    positives = _positives_by_user(_df_interacoes())
    assert positives == {1: {0, 1}, 2: {2}}


def test_negativos_nao_incluem_itens_conhecidos():
    rng = np.random.default_rng(42)
    positives = {1: {0, 1}}
    known = {1: {0, 1, 2}}  # item 2 é positivo em outro split
    neg_df = _generate_negatives(positives, known, n_items=10, neg_ratio=4, rng=rng)

    assert (neg_df["label"] == 0).all()
    assert not {0, 1, 2} & set(neg_df["item_id"])


def test_negativos_respeitam_proporcao():
    rng = np.random.default_rng(42)
    positives = {1: {0, 1}}
    neg_df = _generate_negatives(positives, positives, 100, neg_ratio=4, rng=rng)
    assert len(neg_df) == 8  # 2 positivos x ratio 4


def test_amostragem_mantem_historico_completo_do_usuario():
    df = pd.DataFrame(
        {
            "userId": [1] * 10 + [2] * 10 + [3] * 2,
            "rating": [5.0] * 22,
        }
    )
    sampled = _sample_active_users(df, sample_size=10, seed=42)

    # usuário 3 fica de fora (menos de 5 interações); quem entra, entra inteiro
    assert 3 not in sampled["userId"].to_numpy()
    for _uid, group in sampled.groupby("userId"):
        assert len(group) == 10


def test_amostragem_respeita_orcamento_de_interacoes():
    df = pd.DataFrame(
        {
            "userId": [1] * 10 + [2] * 10 + [3] * 10,
            "rating": [5.0] * 30,
        }
    )
    sampled = _sample_active_users(df, sample_size=20, seed=42)
    assert len(sampled) <= 20
    assert len(sampled) > 0
