from yamlattributes import YamlAttributes
from typing import Optional, Union


class ModelConfig(YamlAttributes):
    yaml_file_path = './config/models.yaml'
    image_size: int
    patch_size: int
    model_dim: int
    depth: int
    heads: int
    mlp_dim: int


class TrainConfig(YamlAttributes):
    batch_size: int
    test_set_batch_size: Optional[int]
    num_classes: int
    epochs: int
    lr: float
    channels: int
    num_workers: int
    data_dir: str
    dataset: str
    log_steps: int
    scheduler: Optional[str]
    drop_last: Optional[bool]
    metrics_debug: Optional[bool]
    alpha_mixup: Optional[float]
    weight_decay: Optional[float]
    nprocs: Optional[int]
    resize_image_size: Optional[int]
    warmup_steps: Optional[int]
    dataset_size: Optional[int]
    test_dataset_size: Optional[int]
    alpha_mixup: Optional[float]
    subset: Optional[float]
    dropout: Optional[float]


class TuneConfig(YamlAttributes):
    yaml_file_path = './config/tune.yaml'
    lr: Optional[Union[list, dict, float]]
    weight_decay: Optional[Union[list, dict, float]]
    dropout: Optional[Union[list, dict, float]]
    subset: Optional[Union[list, dict, float]]
    batch_size: Optional[Union[list, dict, int]]
    warmup_steps: Optional[Union[list, dict, int]]
