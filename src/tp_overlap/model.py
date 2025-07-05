import math
from dataclasses import dataclass

import torch
from torch import nn

from tp_overlap.config import ModelConfig


@dataclass
class KVCache:
    key: torch.Tensor
    value: torch.Tensor


def rms_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + 1e-5).to(x.dtype) * weight


def causal_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, *, causal: bool
) -> torch.Tensor:
    scores = query.float() @ key.float().transpose(-1, -2) / math.sqrt(query.shape[-1])
    if causal:
        length = query.shape[-2]
        mask = torch.ones(length, length, dtype=torch.bool, device=query.device).triu(1)
        scores = scores.masked_fill(mask, float("-inf"))
    probabilities = scores.softmax(dim=-1).to(query.dtype)
    return probabilities @ value


class TinyCausalLM(nn.Module):
    """Single-layer decoder used as the unsharded numerical reference."""

    def __init__(self, config: ModelConfig | None = None, seed: int = 7) -> None:
        super().__init__()
        config = config or ModelConfig()
        self.config = config
        generator = torch.Generator().manual_seed(seed)

        def parameter(*shape: int) -> nn.Parameter:
            tensor = torch.randn(*shape, generator=generator, dtype=torch.float32) * 0.15
            return nn.Parameter(tensor, requires_grad=False)

        self.embedding = nn.Parameter(
            torch.randn(config.vocab_size, config.hidden_size, generator=generator) * 0.2,
            requires_grad=False,
        )
        self.attention_norm = parameter(config.hidden_size)
        self.mlp_norm = parameter(config.hidden_size)
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.output = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.gate = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        with torch.no_grad():
            for module in (self.qkv, self.output, self.gate, self.up, self.down, self.lm_head):
                module.weight.copy_(
                    torch.randn(module.weight.shape, generator=generator) * 0.15
                )
            self.attention_norm.fill_(1.0)
            self.mlp_norm.fill_(1.0)

    def _attention(
        self, hidden: torch.Tensor, cache: KVCache | None
    ) -> tuple[torch.Tensor, KVCache]:
        config = self.config
        tokens = hidden.shape[0]
        query, key, value = self.qkv(hidden).chunk(3, dim=-1)

        def heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(tokens, config.num_heads, config.head_dim).transpose(0, 1)

        query, key, value = heads(query), heads(key), heads(value)
        if cache is not None:
            key = torch.cat((cache.key, key), dim=1)
            value = torch.cat((cache.value, value), dim=1)
        attended = causal_attention(query, key, value, causal=cache is None)
        attended = attended.transpose(0, 1).reshape(tokens, config.hidden_size)
        return self.output(attended), KVCache(key, value)

    def _mlp(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.down(torch.nn.functional.silu(self.gate(hidden)) * self.up(hidden))

    @torch.no_grad()
    def prefill(self, token_ids: list[int]) -> tuple[torch.Tensor, KVCache]:
        if not token_ids:
            raise ValueError("prefill requires at least one token")
        ids = torch.tensor(token_ids, dtype=torch.long, device=self.embedding.device)
        hidden = self.embedding[ids]
        attention, cache = self._attention(rms_norm(hidden, self.attention_norm), None)
        hidden = hidden + attention
        hidden = hidden + self._mlp(rms_norm(hidden, self.mlp_norm))
        return self.lm_head(hidden), cache

    @torch.no_grad()
    def decode(self, token_id: int, cache: KVCache) -> tuple[torch.Tensor, KVCache]:
        ids = torch.tensor([token_id], dtype=torch.long, device=self.embedding.device)
        hidden = self.embedding[ids]
        attention, cache = self._attention(rms_norm(hidden, self.attention_norm), cache)
        hidden = hidden + attention
        hidden = hidden + self._mlp(rms_norm(hidden, self.mlp_norm))
        return self.lm_head(hidden)[0], cache
