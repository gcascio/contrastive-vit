# xla_mock overrides the xla module imports on non tpu machines
import xla_mock # noqa

import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

import torch

from vit.vit import ViT
from train.train import train
from train_utils.datasets.dataset_loader import get_dataset
from utils.args import get_args
from config.config import ModelConfig, TrainConfig
from utils.logging import get_log_base_path
from utils.summary_writer import SummaryWriter
from utils.seed import seed_all
from os.path import join


args = get_args()

ModelConfig.init(yaml_section=args.vit_model)
TrainConfig.init(yaml_file_path=args.train_config)


def map_fn(_, model, train_config: TrainConfig, model_config: ModelConfig):
    torch.set_default_tensor_type('torch.FloatTensor')
    base_path = get_log_base_path(
        train_config.dataset,
        image_size=model_config.image_size,
        model=args.vit_model,
        dir='train',
    )

    train_dataset, test_dataset = get_dataset(
        train_config,
        model_config.image_size,
    )

    writer = None
    if xm.is_master_ordinal():
        writer = SummaryWriter(log_dir=base_path)

    _, best_result = train(
        model,
        train_config,
        train_dataset,
        test_dataset,
        writer,
        model_config=model_config,
    )

    if writer:
        writer.add_hparams(
            {
                **TrainConfig.to_dict(),
                **ModelConfig.to_dict(),
            },
            {
                'Accuracy/train': None,
            },
        )

        writer.flush()
        writer.close()

    xm.save(best_result, join(base_path, 'best_model.pth'))


if __name__ == "__main__":
    seed_all(3)

    vision_transformer = ViT(
        image_size=ModelConfig.image_size,
        channels=TrainConfig.channels,
        patch_size=ModelConfig.patch_size,
        num_classes=TrainConfig.num_classes,
        model_dim=ModelConfig.model_dim,
        depth=ModelConfig.depth,
        heads=ModelConfig.heads,
        mlp_dim=ModelConfig.mlp_dim,
        dropout=TrainConfig.dropout,
    )

    xmp.spawn(
        map_fn,
        args=(vision_transformer, TrainConfig, ModelConfig),
        nprocs=TrainConfig.nprocs or 1,
    )
