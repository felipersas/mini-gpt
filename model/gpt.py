"""GPT decoder-only: embeddings, blocos Transformer, LayerNorm final e LM head."""

import torch
from torch import nn

from config import ModelConfig
from model.embeddings import Embeddings
from model.layer_norm import LayerNorm
from model.lm_head import LMHead
from model.transformer_block import TransformerBlock


class GPT(nn.Module):
    """IDs [batch, seq_len] -> logits [batch, seq_len, vocab_size].

    `vocab_size` vem do tokenizer, não da config.
    """

    def __init__(self, config: ModelConfig, vocab_size: int) -> None:
        super().__init__()
        self.config = config
        self.embeddings = Embeddings(
            vocab_size, config.context_length, config.d_model, config.dropout
        )
        self.blocks = nn.ModuleList(
            TransformerBlock(
                config.d_model,
                config.num_heads,
                config.d_ff,
                config.dropout,
                config.num_layers,
                attention_backend=config.attention_backend,
            )
            for _ in range(config.num_layers)
        )
        # No pre-norm o residual stream nunca é normalizado; esta LayerNorm padroniza a saída.
        self.ln_final = LayerNorm(config.d_model)
        self.lm_head = LMHead(config.d_model, vocab_size)
        if config.tie_weights:
            # Mesma matriz [vocab_size, d_model]: a linha i representa o token i na entrada
            # (lookup) e pontua o token i na saída (produto escalar).
            self.lm_head.proj.weight = self.embeddings.token_embedding.weight

    def hidden_states(self, ids: torch.Tensor) -> torch.Tensor:
        """IDs [batch, seq_len] -> estados finais [batch, seq_len, d_model]."""
        h = self.embeddings(ids)  # [batch, seq_len, d_model]
        for block in self.blocks:
            h = block(h)  # [batch, seq_len, d_model]
        return self.ln_final(h)  # [batch, seq_len, d_model]

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        # ids: [batch, seq_len]
        return self.lm_head(self.hidden_states(ids))  # [batch, seq_len, vocab_size]

    def num_parameters(self) -> int:
        # parameters() não repete tensores compartilhados: com tying, a matriz conta uma vez.
        return sum(p.numel() for p in self.parameters())
