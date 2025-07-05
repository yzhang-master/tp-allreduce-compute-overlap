from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    hidden_size: int = 32
    intermediate_size: int = 64
    num_heads: int = 4
    vocab_size: int = 97

    def __post_init__(self) -> None:
        if min(self.hidden_size, self.intermediate_size, self.num_heads, self.vocab_size) <= 0:
            raise ValueError("model dimensions must be positive")
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    def validate_world_size(self, world_size: int) -> None:
        if world_size <= 0:
            raise ValueError("world_size must be positive")
        if self.num_heads % world_size or self.intermediate_size % world_size:
            raise ValueError("world_size must divide num_heads and intermediate_size")
        if self.hidden_size % world_size:
            raise ValueError("world_size must divide hidden_size for sliced reductions")
