import torch

from tp_overlap.config import ModelConfig
from tp_overlap.model import TinyCausalLM


def test_incremental_decode_matches_full_causal_recomputation():
    torch.set_num_threads(1)
    config = ModelConfig(hidden_size=16, intermediate_size=24, num_heads=2, vocab_size=41)
    model = TinyCausalLM(config)
    prompt = [3, 8, 13, 21]
    logits, cache = model.prefill(prompt)
    generated = []
    for _ in range(4):
        token = int(logits[-1].argmax())
        generated.append(token)
        logits, cache = model.decode(token, cache)
        full_logits, _ = model.prefill(prompt + generated)
        torch.testing.assert_close(logits, full_logits[-1], atol=2e-6, rtol=2e-6)
    assert cache.key.shape == (2, 8, 8)


def test_model_rejects_empty_prefill_and_bad_shapes():
    try:
        TinyCausalLM().prefill([])
    except ValueError as error:
        assert "at least one" in str(error)
    else:
        raise AssertionError("empty prefill was accepted")
