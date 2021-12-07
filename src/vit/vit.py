import torch
import math

from torch import nn
from typing import Union
from argguard import ArgGuard
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import Tensor
from scipy.ndimage import zoom

from vit.transformer import TransformerEncoder


def ensure_tuple(x):
    return x if isinstance(x, tuple) else (x, x)


class ViT(nn.Module):
    def __init__(
        self,
        *,
        image_size: Union[int, 'tuple[int]'],
        channels: int = 3,
        patch_size: Union[int, 'tuple[int]'],
        num_classes: int,
        model_dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        dropout: float = 0.,
        with_mlp_head: bool = True,
        expose_attn: bool = False,
    ):
        super().__init__()
        self.expose_attn = expose_attn

        image_height, image_width = ensure_tuple(image_size)
        patch_height, patch_width = ensure_tuple(patch_size)

        self.patch_embedding = FullEmbedding(
            (image_height, image_width, channels),
            (patch_height, patch_width),
            model_dim,
        )

        self.dropout = nn.Dropout(dropout)

        self.transformer = TransformerEncoder(
            model_dim=model_dim,
            depth=depth,
            attention_heads=heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            expose_attn=expose_attn,
        )

        if with_mlp_head:
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(model_dim),
                nn.Linear(model_dim, num_classes)
            )
        else:
            self.mlp_head = nn.Identity()

    def forward(self, img: Tensor):
        weights = None

        x = self.patch_embedding(img)

        x = self.dropout(x)

        x, weights = self.transformer(x)

        # Get state of the class token
        x = x[:, 0]

        x = self.mlp_head(x)

        if not self.expose_attn:
            return x

        return x, weights


class PatchEmbedding(ArgGuard, nn.Module):
    def __init__(self, image_dimension: 'tuple[int]',
                 patch_size: 'tuple[int]', embed_dim: int):
        super().__init__()
        self._assert_args(locals())

        *_, channels = image_dimension

        # Instead of splitting the input image in to patches and then
        # running them through a linear projection both steps can be combined
        # in a single convolution layer which should bring a performance gain
        self.conv_layer = nn.Conv2d(
            channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # b: batch size
        # e: embedding dimension
        # n_pv: number of patches along the vertical axis
        # n_ph: number of patches along the horizontal axis
        self.rearrange = Rearrange('b e n_pv n_ph -> b (n_pv n_ph) e')

    def forward(self, img: Tensor):
        out = self.conv_layer(img)
        out = self.rearrange(out)
        return out

    def _arg_guard(self):
        image_dimension, patch_size = self._get_required_args(
            'image_dimension',
            'patch_size',
        )

        *image_2D_dimension, _ = image_dimension

        assert (
            len(image_2D_dimension) == len(patch_size)
        ), 'Image and patch size must have the same dimension'

        assert all(
            d_i % d_p == 0 for d_i, d_p in zip(image_2D_dimension, patch_size)
        ), 'The image size must be divisible by the patch size.'


class PositionalEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embed_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_embeddings, embed_dim))

    def forward(self, x: Tensor):
        out = x + self.pos_embedding
        return out

    # When fine tuning at higher resolution while keeping the patch size equal
    # the learned positional embedding loses its meaning. To counteract this
    # effect 2D interpolation is applied to the learned position embeddings
    # as described in https://arxiv.org/abs/2010.11929
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        pos_embedding_key = prefix + 'pos_embedding'
        loaded_pos_embedding = state_dict[pos_embedding_key][0]

        # The -1 accounts for the classification token
        num_patch_embeddings = self.num_embeddings - 1
        loaded_num_patch_embeddings = loaded_pos_embedding.shape[0] - 1

        if num_patch_embeddings == loaded_num_patch_embeddings:
            loaded_patch_size = math.isqrt(loaded_num_patch_embeddings)
            patch_size = math.isqrt(num_patch_embeddings)

            assert (
                loaded_num_patch_embeddings == loaded_patch_size ** 2
                and num_patch_embeddings == patch_size ** 2
            ), 'Only the loading of models with square patches is supported'

            patch_ratio = patch_size / loaded_patch_size

            loaded_token_pos_embedding = loaded_pos_embedding[:1, :]
            loaded_patch_pos_embeddings = loaded_pos_embedding[1:, :]

            # p_h: patches height
            # p_w: patches width
            # e: embedding dimension
            loaded_patch_pos_grid = rearrange(
                loaded_patch_pos_embeddings,
                '(p_h p_w) e -> p_h p_w e',
                p_h=loaded_patch_size,
            )

            # Interpolate the learned position embeddings using spline
            # interpolation as  presented in https://github.com/google-research/vision_transformer/blob/39c905d2caf96a4306c9d78f05df36ddb3eb8ecb/vit_jax/checkpoint.py#L173 # noqa: E501
            interpolated_patch_pos_grid = torch.from_numpy(zoom(
                loaded_patch_pos_grid,
                (patch_ratio, patch_ratio, 1),
                order=1,
            ))

            # p_h: patches height
            # p_w: patches width
            # e: embedding dimension
            interpolated_patch_pos_emb = rearrange(
                interpolated_patch_pos_grid,
                'p_h p_w e -> (p_h p_w) e',
            )

            # Prepend the interpolated patch position embeddings with the
            # position embedding of the token and write the result back into
            # the state dict
            state_dict[pos_embedding_key] = torch.cat(
                (loaded_token_pos_embedding, interpolated_patch_pos_emb),
            ).unsqueeze(0)

        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)


class FullEmbedding(nn.Module):
    def __init__(self, image_dimension: 'tuple[int]',
                 patch_size: 'tuple[int]', embed_dim: int):
        super().__init__()
        num_patches = self.__calculate_number_of_patches(
            image_dimension, patch_size)

        self.patch_embedding = PatchEmbedding(
            image_dimension,
            patch_size,
            embed_dim,
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # The +1 accounts for the prepended cls token
        self.positional_embedding = PositionalEmbedding(
            num_patches + 1, embed_dim)

    def forward(self, x: Tensor):
        batch_size, *_ = x.shape

        patch_embedding = self.patch_embedding(x)

        cls_tokens = self.cls_token.repeat_interleave(batch_size, 0)
        full_patch_embedding = torch.cat([cls_tokens, patch_embedding], dim=1)

        positional_embedding = self.positional_embedding(full_patch_embedding)
        return positional_embedding

    def __calculate_number_of_patches(
            self, image_dimension: 'tuple[int]', patch_size: 'tuple[int]'):
        image_height, image_width, _ = image_dimension
        patch_height, patch_width = patch_size

        num_patches_vertical = image_height // patch_height
        num_patches_horizontal = image_width // patch_width

        num_patches = num_patches_vertical * num_patches_horizontal
        return num_patches
