import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel

from config import Config


class AttentionPooling(nn.Module):
    """
    Learns a query vector to compute a weighted average over token representations,
    capturing richer context than a plain CLS token or mean pool.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        # Score each token; mask padding positions with -inf before softmax
        scores = self.attn(hidden_states).squeeze(-1)  # (B, L)
        scores = scores.masked_fill(attention_mask == 0, float("-inf"))
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)  # (B, L, 1)
        return (hidden_states * weights).sum(dim=1)  # (B, H)


class MatryoshkaProjection(nn.Module):
    """
    Projects encoder output to EMBEDDING_DIM and supports Matryoshka
    Representation Learning (MRL) by exposing sub-vectors at multiple granularities.

    Reference: Kusupati et al., 2022 — "Matryoshka Representation Learning"
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def encode_at_dim(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        """Return L2-normalized embedding truncated to `dim` dimensions."""
        full = self.forward(x)
        return F.normalize(full[:, :dim], p=2, dim=-1)


class RDModel(nn.Module):
    """
    Dual-encoder reverse dictionary model.

    Architecture:
      - Shared BERT backbone with independent fine-tuning paths
      - Attention pooling over all token representations (replaces CLS-only)
      - Matryoshka projection head for multi-granularity embeddings
    """

    def __init__(self, model_name: str = Config.MODEL_NAME):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        hidden = self.config.hidden_size  # 768 for bert-base

        self.bert = AutoModel.from_pretrained(model_name)
        self.pool = AttentionPooling(hidden)
        self.proj = MatryoshkaProjection(hidden, Config.EMBEDDING_DIM)

    def _encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        dim: int = Config.EMBEDDING_DIM,
    ) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.pool(outputs.last_hidden_state, attention_mask)
        return self.proj.encode_at_dim(pooled, dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        dim: int = Config.EMBEDDING_DIM,
    ) -> torch.Tensor:
        return self._encode(input_ids, attention_mask, dim)

    def encode_word(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        dim: int = Config.EMBEDDING_DIM,
    ) -> torch.Tensor:
        """Encode a target word token sequence."""
        return self._encode(input_ids, attention_mask, dim)

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        dim: int = Config.EMBEDDING_DIM,
    ) -> torch.Tensor:
        """Encode a definition / query token sequence."""
        return self._encode(input_ids, attention_mask, dim)

    def encode_all_mrl_dims(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[int, torch.Tensor]:
        """
        Return embeddings at every MRL dimension in a single forward pass.
        Used during training to compute multi-scale loss.
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.pool(outputs.last_hidden_state, attention_mask)
        full_proj = self.proj(pooled)

        return {
            dim: F.normalize(full_proj[:, :dim], p=2, dim=-1) for dim in Config.MRL_DIMS
        }
