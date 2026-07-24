from dataclasses import dataclass, asdict
from pathlib import Path
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_CONFIG = REPO_ROOT / "configs" / "model.yaml"


@dataclass
class DataConfig:
    horizon: int = 5


@dataclass
class EvalConfig:
    n_splits: int = 5
    min_train_size: int = 252
    val_frac: float = 0.2


@dataclass
class ModelConfig:
    seq_len: int = 30
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.2


@dataclass
class TrainConfig:
    epochs: int = 50
    lr: float = 1.0e-3
    patience: int = 10
    batch_size: int = 32
    seed: int = 42


@dataclass
class Config:
    data: DataConfig
    eval: EvalConfig
    model: ModelConfig
    train: TrainConfig

    def flatten(self) -> dict:
        nested_dict = asdict(self)
        flat_dict = {}

        for key in nested_dict:
            for sub_key in nested_dict[key]:
                flat_dict[f"{key}.{sub_key}"] = nested_dict[key][sub_key]

        return flat_dict


def load_config(path: Path = DEFAULT_MODEL_CONFIG) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return Config(
        data=DataConfig(**raw["data"]),
        eval=EvalConfig(**raw["eval"]),
        model=ModelConfig(**raw["model"]),
        train=TrainConfig(**raw["train"]),
    )
