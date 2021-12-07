from vit.vit import ViT
from utils.args import get_args
from config.config import ModelConfig, TrainConfig
from utils.seed import seed_all


args = get_args()

ModelConfig.init(yaml_section=args.vit_model)
TrainConfig.init(yaml_file_path=args.train_config)


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
        expose_attn=True,
    )

    params = vision_transformer.parameters()

    total_trainable_params = sum(p.numel() for p in params if p.requires_grad)

    print(f'== Num Params: {total_trainable_params} ==')
