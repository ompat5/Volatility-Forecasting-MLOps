"""Typed configuration for the global volatility system."""

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GLOBAL_MODEL_CONFIG = REPO_ROOT / "configs" / "global_model.yaml"


@dataclass
class DataConfig:
    horizon: int = 5


@dataclass
class GlobalEvalConfig:
    n_splits: int = 5
    min_train_size: int = 252
    val_frac: float = 0.2
    holdout_size: int = 252


@dataclass
class GlobalModelConfig:
    seq_len: int = 30
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.2
    embedding_dim: int = 8
    target_transform: str = "log"


@dataclass
class GlobalTrainConfig:
    epochs: int = 50
    lr: float = 1.0e-3
    patience: int = 10
    batch_size: int = 512
    dates_per_batch: int = 4
    seed: int = 42


@dataclass
class BaselineConfig:
    ewma_span: float = 32.0
    garch_update: str = "fixed_parameters"


@dataclass
class GlobalConfig:
    data: DataConfig
    eval: GlobalEvalConfig
    model: GlobalModelConfig
    train: GlobalTrainConfig
    baselines: BaselineConfig

    def flatten(self) -> dict[str, object]:
        """Return dotted configuration keys suitable for experiment tracking."""
        return {
            f"{section}.{key}": value
            for section, values in asdict(self).items()
            for key, value in values.items()
        }


def load_global_config(
    path: Path = DEFAULT_GLOBAL_MODEL_CONFIG,
) -> GlobalConfig:
    """Load the sole model configuration used by this repository."""
    with path.open() as config_file:
        raw = yaml.safe_load(config_file)
    config = GlobalConfig(
        data=DataConfig(**raw["data"]),
        eval=GlobalEvalConfig(**raw["eval"]),
        model=GlobalModelConfig(**raw["model"]),
        train=GlobalTrainConfig(**raw["train"]),
        baselines=BaselineConfig(**raw["baselines"]),
    )
    if config.model.target_transform != "log":
        raise ValueError("Global model target_transform must be 'log'")
    if config.baselines.garch_update != "fixed_parameters":
        raise ValueError("Global GARCH update mode must be 'fixed_parameters'")
    return config
