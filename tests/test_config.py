import pytest

from tp_overlap.config import ModelConfig


def test_tensor_parallel_shape_constraints():
    config = ModelConfig(hidden_size=32, intermediate_size=64, num_heads=4)
    config.validate_world_size(4)
    with pytest.raises(ValueError, match="world_size"):
        config.validate_world_size(3)
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(hidden_size=30, num_heads=4)
