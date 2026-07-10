import dataclasses

import pytest

from recsys.training.experiments import _CONFIGS, _save_best
from recsys.training.registry import _load_best_run
from recsys.training.train import TrainConfig
from recsys.training.trainer import EarlyStopping


def test_early_stopping_para_apos_patience_sem_melhora():
    early = EarlyStopping(patience=2)
    assert not early.should_stop(0.5)
    assert not early.should_stop(0.5)  # 1ª época sem melhora
    assert early.should_stop(0.5)  # 2ª época sem melhora → para


def test_early_stopping_reseta_contador_quando_melhora():
    early = EarlyStopping(patience=2)
    assert not early.should_stop(0.5)
    assert not early.should_stop(0.5)
    assert not early.should_stop(0.3)  # melhorou → zera o contador
    assert not early.should_stop(0.3)
    assert early.should_stop(0.3)


def test_early_stopping_ignora_melhora_menor_que_min_delta():
    early = EarlyStopping(patience=1, min_delta=1e-2)
    assert not early.should_stop(0.5)
    assert early.should_stop(0.499)  # melhora abaixo do min_delta


def test_train_config_e_imutavel():
    config = TrainConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.lr = 0.1


def test_experimentos_tem_pelo_menos_tres_configs_distintas():
    assert len(_CONFIGS) >= 3
    nomes = {config.run_name for config in _CONFIGS}
    assert len(nomes) == len(_CONFIGS)


def test_best_run_roundtrip_entre_experimentos_e_registry(tmp_path):
    best = {"run_name": "ncf-emb32", "run_id": "abc123", "best_val_loss": 0.42}
    _save_best(best, tmp_path)
    assert _load_best_run(tmp_path) == best


def test_registry_exige_experimentos_executados(tmp_path):
    with pytest.raises(FileNotFoundError, match="experiments"):
        _load_best_run(tmp_path)
