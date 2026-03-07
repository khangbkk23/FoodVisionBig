import yaml
import sys
from dataclasses import dataclass, field
from typing import Union

@dataclass
class DatasetConfig:
    name: str = "food101"
    data_dir: str = "data/"
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    download: bool = True

@dataclass
class TrainingConfig:
    epochs: int = 50
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    seed: int = 42
    device: str = "cuda"
    checkpoint_dir: str = "model_checkpoints/"
    early_stopping_patience: int = 8

@dataclass
class CustomCNNConfig:
    num_classes: int = 101
    model_name: str = "custom_cnn"
    in_channels: int = 3
    base_filters: int = 32
    dropout_rate: float = 0.5

@dataclass
class EfficientNetConfig:
    num_classes: int = 101
    model_name: str = "efficientnet_b0"
    pretrained: bool = True
    fine_tune: bool = False
    unfreeze_layers: int = 20

@dataclass
class ProjectConfig:
    system_active_model: str
    dataset: DatasetConfig
    training: TrainingConfig
    custom_cnn: CustomCNNConfig
    efficientnet: EfficientNetConfig

    @property
    def active_model_config(self) -> Union[CustomCNNConfig, EfficientNetConfig]:
        if self.system_active_model == "custom_cnn":
            return self.custom_cnn
        elif self.system_active_model == "efficientnet":
            return self.efficientnet
        else:
            raise ValueError(f"{self.system_active_model} is not supported")

def load_config(yaml_path: str = "conf/config.yaml") -> ProjectConfig:
    try:
        with open(yaml_path, "r") as f:
            raw_cfg = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Cannot found YAML file at {yaml_path}")
        sys.exit(1)
    except yaml.YAMLError as exc:
        print(f"Errors are in {exc}")
        sys.exit(1)


    active_model = raw_cfg.get("system", {}).get("active_model", "efficientnet")
    dataset_cfg = DatasetConfig(**raw_cfg.get("dataset", {}))
    training_cfg = TrainingConfig(**raw_cfg.get("training", {}))
    custom_cnn_cfg = CustomCNNConfig(**raw_cfg.get("custom_cnn", {}))
    efficientnet_cfg = EfficientNetConfig(**raw_cfg.get("efficientnet", {}))

    return ProjectConfig(
        system_active_model=active_model,
        dataset=dataset_cfg,
        training=training_cfg,
        custom_cnn=custom_cnn_cfg,
        efficientnet=efficientnet_cfg
    )