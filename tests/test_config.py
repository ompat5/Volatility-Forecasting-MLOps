import textwrap

import pytest

from src.config import (
    BaselineConfig,
    DataConfig,
    GlobalConfig,
    GlobalEvalConfig,
    GlobalModelConfig,
    GlobalTrainConfig,
    load_global_config,
)


def test_load_global_config_returns_the_only_supported_configuration():
    config = load_global_config()

    assert isinstance(config, GlobalConfig)
    assert isinstance(config.data, DataConfig)
    assert isinstance(config.eval, GlobalEvalConfig)
    assert isinstance(config.model, GlobalModelConfig)
    assert isinstance(config.train, GlobalTrainConfig)
    assert isinstance(config.baselines, BaselineConfig)
    assert config.data.horizon == 5
    assert config.eval.holdout_size == 252
    assert config.model.embedding_dim == 8
    assert config.train.dates_per_batch == 4
    assert isinstance(config.train.lr, float)
    assert config.flatten()["model.embedding_dim"] == 8


def test_unknown_configuration_key_raises(tmp_path):
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
              holdout_size: 252
            model:
              seq_len: 30
              hidden_size: 64
              num_layers: 1
              dropout: 0.2
              embedding_dim: 8
              target_transform: log
            train:
              epochs: 50
              lr: 1.0e-3
              patience: 10
              batch_size: 512
              dates_per_batch: 4
              seed: 42
            baselines:
              ewma_span: 32.0
              garch_update: fixed_parameters
            """
        )
    )

    with pytest.raises(TypeError):
        load_global_config(bad)
