"""A compact BERT-style encoder with no pretrained-model dependency."""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class BertStyleConfig:
    vocab_size: int = 128
    hidden_size: int = 32
    num_heads: int = 4
    intermediate_size: int = 64
    num_layers: int = 2
    max_positions: int = 64


class SelfAttention(nn.Module):
    def __init__(self, config: BertStyleConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.qkv_projection = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.output_projection = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch, sequence, hidden_size = hidden.shape
        qkv = self.qkv_projection(hidden).reshape(
            batch, sequence, 3, self.num_heads, self.head_dim
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        scores = query @ key.transpose(-2, -1) * self.head_dim**-0.5
        visible = (attention_mask != 0)[:, None, None, :]
        scores = scores.masked_fill(~visible, -10000.0)
        context = torch.softmax(scores, dim=-1) @ value
        context = context.transpose(1, 2).reshape(batch, sequence, hidden_size)
        return self.output_projection(context)


class EncoderLayer(nn.Module):
    def __init__(self, config: BertStyleConfig):
        super().__init__()
        self.attention = SelfAttention(config)
        self.attention_norm = nn.LayerNorm(config.hidden_size)
        self.input_projection = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output_projection = nn.Linear(config.intermediate_size, config.hidden_size)
        self.output_norm = nn.LayerNorm(config.hidden_size)
        self.activation = nn.GELU()

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.attention_norm(hidden + self.attention(hidden, attention_mask))
        feed_forward = self.output_projection(self.activation(self.input_projection(hidden)))
        return self.output_norm(hidden + feed_forward)


class BertStyleEncoder(nn.Module):
    """BERT-like embeddings and post-norm encoder layers returning token hidden states."""

    def __init__(self, config: BertStyleConfig | None = None):
        super().__init__()
        self.config = config or BertStyleConfig()
        self.token_embeddings = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.position_embeddings = nn.Embedding(
            self.config.max_positions, self.config.hidden_size
        )
        self.embedding_norm = nn.LayerNorm(self.config.hidden_size)
        self.layers = nn.ModuleList(
            EncoderLayer(self.config) for _ in range(self.config.num_layers)
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        hidden = self.token_embeddings(input_ids) + self.position_embeddings(positions)[None, :, :]
        hidden = self.embedding_norm(hidden)
        for layer in self.layers:
            hidden = layer(hidden, attention_mask)
        return hidden * attention_mask.unsqueeze(-1).to(hidden.dtype)
