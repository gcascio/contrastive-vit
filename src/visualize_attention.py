import glob
import math
import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2

from vit.vit import ViT
from config.config import ModelConfig, TrainConfig
from utils.seed import seed_all
from PIL import Image
from torchvision import transforms


def attention_rollout(attention_matrix: torch.Tensor, top_attention_cut=None):
    attention_map_size = attention_matrix[0].size(-1)
    att_rollout = torch.eye(attention_map_size)
    identity_matrix = torch.eye(attention_map_size)

    with torch.no_grad():
        for attention in attention_matrix:
            # average over all attention heads
            attention_mean = attention.mean(axis=1)

            # Only consider the 'top_attention_cut' highest
            # attentions and set the rest to zero. Empirically
            # this gives a clearer depiction of the attention
            # when used as a mask on images.
            if top_attention_cut:
                flattened_att = attention_mean.flatten()
                n_cuts = int(flattened_att.size(0) * (1 - top_attention_cut))
                _, cut_indices = flattened_att.topk(n_cuts, largest=False)
                flattened_att[cut_indices] = 0

            raw_attention = (attention_mean + identity_matrix) / 2

            att_rollout = torch.matmul(raw_attention, att_rollout)

    return att_rollout


def apply_mask_to_image(img, mask):
    n_patches = math.isqrt(mask.size(-1))

    mask = mask.reshape(n_patches, n_patches)
    mask = mask / mask.max()

    mask = mask.numpy()
    mask = cv2.resize(mask, img.size)
    mask = np.expand_dims(mask, -1)

    masked_img = (img * mask).astype("uint8")
    return masked_img


def load_file_paths(root: str, file_ending='JPEG', size=1):
    root = os.path.expanduser(root)
    files = glob.glob(f'{root}/**/*.{file_ending}', recursive=True)
    return np.random.choice(files, size)


if __name__ == "__main__":
    seed_all(2)
    parser = argparse.ArgumentParser(description='Training Config')
    parser.add_argument('--checkpoint_path', metavar='PATH',
                        help='path to checkpoint')
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint_path)
    model_config = checkpoint['model_config']
    train_config = checkpoint['train_config']
    model_state_dict = checkpoint['model_state_dict']

    ModelConfig.init(config_dict=model_config)
    TrainConfig.init(config_dict=train_config)

    setattr(TrainConfig, 'batch_size', 1)
    setattr(TrainConfig, 'test_set_batch_size', 1)

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

    vision_transformer.load_state_dict(model_state_dict)
    vision_transformer.eval()

    norm = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    transform = transforms.Compose([
        transforms.ToTensor(),
        norm,
    ])

    data_base_path = '~/HHU/Bachelor/datasets/tiny-imagenet-200/val/'  # noqa: E501
    rows = 10

    fig, ax = plt.subplots(nrows=rows, ncols=7, figsize=(8, 10),
                           gridspec_kw={'width_ratios': [3, 3, 3, 1, 3, 3, 3]})

    num_images = 2 * rows
    img_paths = load_file_paths(data_base_path, size=num_images)

    for k, img_path in enumerate(img_paths):
        i = k % rows
        j = (k // rows) * 4

        img = Image.open(img_path)

        x = transform(img).unsqueeze(0)

        _, att_mat = vision_transformer(x)

        att_rollout = attention_rollout(att_mat, top_attention_cut=1.)
        att_rollout_cut = attention_rollout(att_mat, top_attention_cut=.1)

        # get the attention from the class token to all positions
        # except to the class token itself
        mask = att_rollout[0, 0, 1:]
        mask_cut = att_rollout_cut[0, 0, 1:]

        masked_img = apply_mask_to_image(img, mask)
        masked_img_cut = apply_mask_to_image(img, mask_cut)

        ax[i][0 + j].imshow(img)
        ax[i][1 + j].imshow(masked_img)
        ax[i][2 + j].imshow(masked_img_cut)

        if j == 4:
            ax[i][3].set_axis_off()

        if i == 0:
            ax[i][0 + j].set_title('Original', fontsize=10, pad=12)
            ax[i][1 + j].set_title('Attn. Rollout', fontsize=10, pad=12)
            ax[i][2 + j].set_title('Top-10 \n Attn. Rollout', fontsize=10)

        ax[i][0 + j].set_xticks([])
        ax[i][0 + j].set_yticks([])
        ax[i][1 + j].set_xticks([])
        ax[i][1 + j].set_yticks([])
        ax[i][2 + j].set_xticks([])
        ax[i][2 + j].set_yticks([])

        label_count = i + (j // 4) * rows + 1
        ax[i][0 + j].set_ylabel(label_count, rotation=0, labelpad=12)

        ax[i][0 + j].set_xmargin(50.)
        ax[i][1 + j].set_xmargin(50.)
        ax[i][2 + j].set_xmargin(50.)

    fig.tight_layout()
    plt.show()
