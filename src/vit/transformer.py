from torch import nn
from torch import Tensor
from einops import rearrange
from argguard import ArgGuard


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.scaling = d_k ** -0.5
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q: Tensor, K: Tensor, V: Tensor):
        weights = self.softmax(Q.matmul(K.transpose(-1, -2)) * self.scaling)
        attention = weights.matmul(V)
        return attention, weights


class MultiHeadAttention(ArgGuard, nn.Module):
    """MultiHeadAttention.

    Efficient multi head attention implementation for the special case
    d_k = d_q = d_v

    Attributes:
        image_dimension: dimension of input images (height, width, channels)
        patch_size: desired patch size (height, width)
        embed_dim: dimension of the patch embeddings
    """
    def __init__(self, model_dim: int, num_heads: int, d_k: int, d_v: int):
        super().__init__()
        self._assert_args(locals())

        self.num_heads = num_heads

        # times 3 to account for q, k and v at the same time
        self.qkv_projection = nn.Linear(model_dim, num_heads * d_k * 3)
        self.scaled_dot_product_attention = ScaledDotProductAttention(d_k)
        self.linear_layer = nn.Linear(num_heads * d_v, model_dim)

    def forward(self, x):
        projected_qkv = self.qkv_projection(x)

        # b: batch size
        # s: sequence length
        # c: number of concatenated vectors
        projected_query, projected_key, projected_value = rearrange(
            projected_qkv, 'b s (h d c) -> c b h s d', c=3, h=self.num_heads)

        out, weights = self.scaled_dot_product_attention(
            projected_query, projected_key, projected_value)

        out = rearrange(out, 'b h s d -> b s (h d)')

        out = self.linear_layer(out)

        return out, weights

    def _arg_guard(self):
        d_k, d_v = self._get_required_args('d_k', 'd_v')

        assert (
            d_k == d_v
        ), 'Dimensions of keys and values have to match'


class MlpLayer(nn.Module):
    def __init__(self, model_dim: int, hidden_dim: int, dropout: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(model_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, model_dim),
        )

    def forward(self, x):
        out = self.layers(x)

        return out


class AttentionLayer(nn.Module):
    def __init__(self, model_dim: int, attention_heads: int):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(
            model_dim,
            attention_heads,
            model_dim // attention_heads,
            model_dim // attention_heads,
        )

    def forward(self, x):
        out, weights = self.multi_head_attention(x)

        return out, weights


class TransformerEcoderLayer(nn.Module):
    def __init__(self, model_dim: int, num_attention_heads: int,
                 mlp_dim: int, dropout: float):
        super().__init__()
        self.first_sublayer = nn.Sequential(
            nn.LayerNorm(model_dim),
            AttentionLayer(model_dim, num_attention_heads),
        )

        self.dropout = nn.Dropout(dropout)

        self.second_sublayer = nn.Sequential(
            nn.LayerNorm(model_dim),
            MlpLayer(model_dim, mlp_dim, dropout),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        first_sublayer_out, weights = self.first_sublayer(x)
        x = self.dropout(x)
        first_sublayer_out += x

        second_sublayer_out = self.second_sublayer(first_sublayer_out)
        second_sublayer_out += first_sublayer_out

        return second_sublayer_out, weights


class TransformerEncoder(nn.Module):
    def __init__(self, model_dim: int, attention_heads: int, mlp_dim: int,
                 depth: int, dropout: float = 0., expose_attn: bool = False):
        super().__init__()
        self.expose_attn = expose_attn
        self.layers = nn.Sequential(*[TransformerEcoderLayer(
            model_dim,
            attention_heads,
            mlp_dim,
            dropout,
        ) for _ in range(depth)])

    def forward(self, x):
        attn_weights = []
        for layer in self.layers:
            x, weights = layer(x)

            if self.expose_attn:
                attn_weights.append(weights)

        return x, attn_weights
