import torch
from torch import nn

from tp_overlap.model import KVCache, TinyCausalLM, causal_attention, rms_norm
from tp_overlap.overlap import OverlapTimeline, row_parallel_gemm


class TensorParallelLM(nn.Module):
    """One decoder layer with head-sharded attention and dimension-sharded MLP."""

    def __init__(
        self,
        reference: TinyCausalLM,
        rank: int,
        world_size: int,
        *,
        output_chunks: int = 2,
        asynchronous: bool = True,
    ) -> None:
        super().__init__()
        config = reference.config
        config.validate_world_size(world_size)
        if not 0 <= rank < world_size:
            raise ValueError("rank is outside the process group")
        if config.hidden_size % output_chunks:
            raise ValueError("output_chunks must divide hidden_size")
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.output_chunks = output_chunks
        self.asynchronous = asynchronous
        self.timelines: list[OverlapTimeline] = []
        self.local_heads = config.num_heads // world_size
        self.local_hidden = self.local_heads * config.head_dim
        self.local_intermediate = config.intermediate_size // world_size

        def copy(tensor: torch.Tensor) -> nn.Parameter:
            return nn.Parameter(tensor.detach().clone(), requires_grad=False)

        self.embedding = copy(reference.embedding)
        self.attention_norm = copy(reference.attention_norm)
        self.mlp_norm = copy(reference.mlp_norm)
        self.lm_head = copy(reference.lm_head.weight)
        head_start = rank * self.local_hidden
        head_end = head_start + self.local_hidden
        query, key, value = reference.qkv.weight.chunk(3, dim=0)
        self.qkv = copy(
            torch.cat(
                (
                    query[head_start:head_end],
                    key[head_start:head_end],
                    value[head_start:head_end],
                ),
                dim=0,
            )
        )
        self.output = copy(reference.output.weight[:, head_start:head_end])
        mlp_start = rank * self.local_intermediate
        mlp_end = mlp_start + self.local_intermediate
        self.gate = copy(reference.gate.weight[mlp_start:mlp_end])
        self.up = copy(reference.up.weight[mlp_start:mlp_end])
        self.down = copy(reference.down.weight[:, mlp_start:mlp_end])

    def _reduce(
        self, local_input: torch.Tensor, weight: torch.Tensor, layer: str, timeline: OverlapTimeline
    ) -> torch.Tensor:
        return row_parallel_gemm(
            local_input,
            weight,
            output_chunks=self.output_chunks,
            asynchronous=self.asynchronous,
            timeline=timeline,
            layer=layer,
        )

    def _attention(
        self, hidden: torch.Tensor, cache: KVCache | None, timeline: OverlapTimeline
    ) -> tuple[torch.Tensor, KVCache]:
        tokens = hidden.shape[0]
        query, key, value = self.qkv_weighted(hidden).chunk(3, dim=-1)

        def heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(tokens, self.local_heads, self.config.head_dim).transpose(0, 1)

        query, key, value = heads(query), heads(key), heads(value)
        if cache is not None:
            key = torch.cat((cache.key, key), dim=1)
            value = torch.cat((cache.value, value), dim=1)
        attended = causal_attention(query, key, value, causal=cache is None)
        attended = attended.transpose(0, 1).reshape(tokens, self.local_hidden)
        output = self._reduce(attended, self.output, "attention", timeline)
        return output, KVCache(key, value)

    def qkv_weighted(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden @ self.qkv.T

    def _forward(
        self, token_ids: list[int], cache: KVCache | None
    ) -> tuple[torch.Tensor, KVCache]:
        timeline = OverlapTimeline()
        self.timelines.append(timeline)
        ids = torch.tensor(token_ids, dtype=torch.long, device=self.embedding.device)
        hidden = self.embedding[ids]
        attention, cache = self._attention(
            rms_norm(hidden, self.attention_norm), cache, timeline
        )
        hidden = hidden + attention
        normalized = rms_norm(hidden, self.mlp_norm)
        activated = torch.nn.functional.silu(normalized @ self.gate.T) * (normalized @ self.up.T)
        hidden = hidden + self._reduce(activated, self.down, "mlp", timeline)
        return hidden @ self.lm_head.T, cache

    @torch.no_grad()
    def prefill(self, token_ids: list[int]) -> tuple[torch.Tensor, KVCache]:
        if not token_ids:
            raise ValueError("prefill requires at least one token")
        return self._forward(token_ids, None)

    @torch.no_grad()
    def decode(self, token_id: int, cache: KVCache) -> tuple[torch.Tensor, KVCache]:
        return self._forward([token_id], cache)
