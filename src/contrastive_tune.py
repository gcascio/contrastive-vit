# xla_mock overrides the xla module imports on non tpu machines
import xla_mock # noqa

import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

import torch

from vit.vit import ViT
from train.train import train
from train_utils.datasets.dataset_loader import get_dataset
from utils.args import get_args
from config.config import ModelConfig, TrainConfig, TuneConfig
from utils.logging import get_log_base_path, log_run
from utils.summary_writer import SummaryWriter
from utils.struct import Struct
from utils.seed import seed_all
from utils.expand import expand_dict
from os.path import join
from train_utils.wrapper.contrastive_learner import SimCLR


args = get_args()

ModelConfig.init(yaml_section=args.vit_model)
TrainConfig.init(yaml_file_path=args.train_config)
TuneConfig.init()


def map_fn(_, model, train_config: TrainConfig, model_config: ModelConfig,
           save_model=False):
    torch.set_default_tensor_type('torch.FloatTensor')
    base_path = get_log_base_path(
        train_config.dataset,
        image_size=model_config.image_size,
        model=args.vit_model,
    )

    train_dataset, test_dataset = get_dataset(
        train_config,
        model_config.image_size,
    )

    writer = None
    if xm.is_master_ordinal():
        writer = SummaryWriter(log_dir=base_path)

    log_run(train_config)

    _, best_result = train(
        model,
        train_config,
        train_dataset,
        test_dataset,
        writer,
        model_returns_loss=True,
    )

    if writer:
        writer.add_hparams(
            {
                **train_config.to_dict(),
                **model_config.to_dict(),
            },
            {
                'Accuracy/train': None,
            },
        )

        writer.flush()
        writer.close()

    if save_model:
        xm.save(best_result, join(base_path, 'best_model.pth'))


if __name__ == "__main__":
    seed_all(3)

    configs = expand_dict(TuneConfig.to_dict())

    def run_train(config=None):
        config_dict = TrainConfig.to_dict()

        if config:
            for key, value in config.items():
                config_dict[key] = value

        train_config = Struct(**config_dict)

        vision_transformer = ViT(
            image_size=ModelConfig.image_size,
            channels=train_config.channels,
            patch_size=ModelConfig.patch_size,
            num_classes=train_config.num_classes,
            model_dim=ModelConfig.model_dim,
            depth=ModelConfig.depth,
            heads=ModelConfig.heads,
            mlp_dim=ModelConfig.mlp_dim,
            dropout=train_config.dropout,
            with_mlp_head=False,
        )

        sim_clr = SimCLR(
            vision_transformer,
            model_output_dim=ModelConfig.model_dim,
            batch_size=train_config.batch_size,
            project_dim=128,
            temperature=0.1,
        )

        xmp.spawn(
            map_fn,
            args=(sim_clr, train_config, ModelConfig),
            nprocs=train_config.nprocs or 1,
        )

    for config in configs:
        run_train(config)
