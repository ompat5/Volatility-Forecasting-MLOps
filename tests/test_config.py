import textwrap

import pytest

from src.config import (
    Config,
    DataConfig,
    EvalConfig,
    BaselineConfig,
    GlobalConfig,
    GlobalEvalConfig,
    GlobalModelConfig,
    GlobalTrainConfig,
    ModelConfig,
    TrainConfig,
    load_config,
    load_global_config,
)


CFG = load_config()
GLOBAL_CFG = load_global_config()


def test_load_config_returns_config():
    assert isinstance(CFG, Config)


def test_subconfigs_have_correct_types():
    assert isinstance(CFG.data, DataConfig)
    assert isinstance(CFG.eval, EvalConfig)
    assert isinstance(CFG.model, ModelConfig)
    assert isinstance(CFG.train, TrainConfig)


def test_values_match_yaml():
    assert CFG.data.horizon == 5
    assert CFG.eval.n_splits == 5
    assert CFG.eval.min_train_size == 252
    assert CFG.eval.val_frac == 0.2
    assert CFG.model.seq_len == 30
    assert CFG.model.hidden_size == 64
    assert CFG.model.num_layers == 1
    assert CFG.model.dropout == 0.2
    assert CFG.train.epochs == 50
    assert CFG.train.patience == 10
    assert CFG.train.batch_size == 32


def test_global_config_is_separate_and_has_holdout():
    assert isinstance(GLOBAL_CFG, GlobalConfig)
    assert isinstance(GLOBAL_CFG.eval, GlobalEvalConfig)
    assert isinstance(GLOBAL_CFG.model, GlobalModelConfig)
    assert isinstance(GLOBAL_CFG.train, GlobalTrainConfig)
    assert isinstance(GLOBAL_CFG.baselines, BaselineConfig)
    assert not hasattr(CFG.eval, "holdout_size")
    assert GLOBAL_CFG.eval.holdout_size == 252
    assert GLOBAL_CFG.model.embedding_dim == 8
    assert GLOBAL_CFG.model.target_transform == "log"
    assert GLOBAL_CFG.train.dates_per_batch == 4
    assert GLOBAL_CFG.baselines.ewma_span == 32.0
    assert GLOBAL_CFG.data.horizon == CFG.data.horizon


def test_lr_is_a_float():
    # Guards the YAML footgun: `1e-3` parses as a string, `1.0e-3` as a float.
    assert isinstance(CFG.train.lr, float)
    assert CFG.train.lr == pytest.approx(1e-3)


def test_unknown_key_raises(tmp_path):
    # A typo in the YAML should fail loudly at load time, not be silently ignored.
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        textwrap.dedent(
            """
            data:
              horizonn: 5
            eval:
              n_splits: 5
              min_train_size: 252
              val_frac: 0.2
            model:
              seq_len: 30
              hidden_size: 64
              num_layers: 1
              dropout: 0.2
            train:
              epochs: 50
              lr: 1.0e-3
              patience: 10
              batch_size: 32
            """
        )
    )
    with pytest.raises(TypeError):
        load_config(bad)
