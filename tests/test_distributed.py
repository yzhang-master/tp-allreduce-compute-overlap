import json
from pathlib import Path

import torch
import torch.multiprocessing as mp

from tp_overlap.config import ModelConfig
from tp_overlap.distributed import destroy_process_group, initialize_process_group
from tp_overlap.model import TinyCausalLM
from tp_overlap.tp_model import TensorParallelLM


def _rank_worker(rank: int, world_size: int, init_file: str, output_dir: str) -> None:
    torch.set_num_threads(1)
    initialize_process_group("gloo", f"file://{init_file}", rank, world_size)
    try:
        config = ModelConfig()
        reference = TinyCausalLM(config)
        prompt = [2, 5, 11, 17, 23]
        result = {}
        for mode, asynchronous in (("overlapped", True), ("sequential", False)):
            model = TensorParallelLM(
                reference, rank, world_size, output_chunks=2, asynchronous=asynchronous
            )
            logits, cache = model.prefill(prompt)
            reference_logits, reference_cache = reference.prefill(prompt)
            max_prefill_error = float((logits - reference_logits).abs().max())
            generated = []
            max_decode_error = 0.0
            for _ in range(4):
                token = int(logits[-1].argmax())
                generated.append(token)
                logits, cache = model.decode(token, cache)
                reference_logits, reference_cache = reference.decode(token, reference_cache)
                max_decode_error = max(
                    max_decode_error, float((logits[-1] - reference_logits).abs().max())
                )
            result[mode] = {
                "prefill_error": max_prefill_error,
                "decode_error": max_decode_error,
                "generated": generated,
                "overlap_seconds": sum(t.overlap_seconds() for t in model.timelines),
                "timeline_events": sum(len(t.events) for t in model.timelines),
            }
        Path(output_dir, f"rank{rank}.json").write_text(json.dumps(result))
    finally:
        destroy_process_group()


def test_two_rank_gloo_inference_correctness_and_overlap_timeline(tmp_path):
    init_file = tmp_path / "gloo_init"
    mp.spawn(
        _rank_worker,
        args=(2, str(init_file), str(tmp_path)),
        nprocs=2,
        join=True,
        daemon=False,
    )
    results = [json.loads((tmp_path / f"rank{rank}.json").read_text()) for rank in range(2)]
    for result in results:
        overlapped = result["overlapped"]
        sequential = result["sequential"]
        assert overlapped["generated"] == sequential["generated"]
        assert overlapped["prefill_error"] < 3e-5
        assert overlapped["decode_error"] < 3e-5
        assert sequential["prefill_error"] < 3e-5
        assert sequential["decode_error"] < 3e-5
        assert overlapped["overlap_seconds"] > 0
        assert overlapped["timeline_events"] == 40
        assert sequential["overlap_seconds"] == 0
