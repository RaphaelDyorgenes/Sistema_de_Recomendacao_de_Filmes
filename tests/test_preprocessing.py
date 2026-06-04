from recsys.preprocessing import (
    IdentityStrategy,
    LabelEncodeStrategy,
    PreprocessingPipeline,
)


def test_identity_nao_altera_valores():
    strategy = IdentityStrategy()
    strategy.fit([3, 1, 2])
    assert strategy.transform([3, 1, 2]) == [3, 1, 2]


def test_label_encode_mapeia_para_intervalo_contiguo():
    strategy = LabelEncodeStrategy()
    strategy.fit([10, 5000, 73])
    assert strategy.transform([10, 5000, 73]) == [0, 2, 1]


def test_label_encode_marca_desconhecido_como_menos_um():
    strategy = LabelEncodeStrategy()
    strategy.fit([10, 20])
    assert strategy.transform([10, 99]) == [0, -1]


def test_label_encode_expoe_tamanho_do_vocabulario():
    strategy = LabelEncodeStrategy()
    strategy.fit([10, 20, 10])
    assert strategy.vocabulary_size == 2


def test_pipeline_encadeia_estrategias():
    pipeline = PreprocessingPipeline([IdentityStrategy(), LabelEncodeStrategy()])
    pipeline.fit([10, 5000, 73])
    assert pipeline.transform([10, 5000, 73]) == [0, 2, 1]
