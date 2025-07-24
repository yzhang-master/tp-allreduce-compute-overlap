import json
import math
import platform
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp

from tp_overlap.config import ModelConfig
from tp_overlap.distributed import destroy_process_group, initialize_process_group
from tp_overlap.model import TinyCausalLM
from tp_overlap.tp_model import TensorParallelLM


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "mean_ms": float(np.mean(values)),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
    }


def _gloo_worker(
    rank: int,
    world_size: int,
    init_file: str,
    output_dir: str,
    decode_tokens: int,
    repeats: int,
) -> None:
    torch.set_num_threads(1)
    initialize_process_group("gloo", f"file://{init_file}", rank, world_size)
    try:
        config = ModelConfig()
        reference = TinyCausalLM(config)
        prompt = [2, 5, 11, 17, 23]
        rank_result = {"rank": rank}
        for mode, asynchronous in (("overlapped", True), ("sequential", False)):
            decode_latencies: list[float] = []
            prefill_latencies: list[float] = []
            overlap_seconds = 0.0
            max_error = 0.0
            generated: list[int] = []
            timeline_sample = []
            for repeat in range(repeats + 1):
                model = TensorParallelLM(
                    reference, rank, world_size, output_chunks=2, asynchronous=asynchronous
                )
                start = time.perf_counter_ns()
                logits, cache = model.prefill(prompt)
                prefill_ms = (time.perf_counter_ns() - start) / 1e6
                reference_logits, reference_cache = reference.prefill(prompt)
                iteration_decode: list[float] = []
                for _ in range(decode_tokens):
                    token = int(logits[-1].argmax())
                    start = time.perf_counter_ns()
                    logits, cache = model.decode(token, cache)
                    iteration_decode.append((time.perf_counter_ns() - start) / 1e6)
                    reference_logits, reference_cache = reference.decode(token, reference_cache)
                    max_error = max(
                        max_error, float((logits[-1] - reference_logits).abs().max())
                    )
                if repeat == 0:
                    continue
                generated = [int(logit.argmax()) for logit in reference_logits.unsqueeze(0)]
                prefill_latencies.append(prefill_ms)
                decode_latencies.extend(iteration_decode)
                overlap_seconds += sum(t.overlap_seconds() for t in model.timelines)
                if not timeline_sample:
                    origin = model.timelines[0].events[0].start_ns
                    timeline_sample = [
                        {
                            **event,
                            "start_ns": int(event["start_ns"]) - origin,
                            "end_ns": int(event["end_ns"]) - origin,
                        }
                        for event in model.timelines[0].as_dicts()
                    ]
            rank_result[mode] = {
                "prefill": _summary(prefill_latencies),
                "inter_token": _summary(decode_latencies),
                "overlap_ms_per_decode": overlap_seconds * 1e3 / (repeats * decode_tokens),
                "max_abs_logit_error": max_error,
                "last_reference_logit_argmax": generated,
                "timeline_sample": timeline_sample,
            }
        Path(output_dir, f"rank{rank}.json").write_text(json.dumps(rank_result))
    finally:
        destroy_process_group()


def run_gloo_benchmark(
    *, world_size: int = 2, decode_tokens: int = 8, repeats: int = 5, output: Path | None = None
) -> dict:
    if world_size < 2 or ModelConfig().num_heads % world_size:
        raise ValueError("CPU benchmark requires world size 2 or 4")
    with tempfile.TemporaryDirectory(prefix="tp-gloo-") as directory:
        init_file = Path(directory, "init")
        mp.spawn(
            _gloo_worker,
            args=(world_size, str(init_file), directory, decode_tokens, repeats),
            nprocs=world_size,
            join=True,
        )
        ranks = [
            json.loads(Path(directory, f"rank{rank}.json").read_text())
            for rank in range(world_size)
        ]

    result = {
        "backend": "gloo",
        "device": "cpu",
        "world_size": world_size,
        "decode_tokens_per_repeat": decode_tokens,
        "repeats": repeats,
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "torch_threads": 1,
        },
        "modes": {},
    }
    for mode in ("sequential", "overlapped"):
        samples = [rank[mode] for rank in ranks]
        result["modes"][mode] = {
            "prefill_rank_mean": _summary([sample["prefill"]["mean_ms"] for sample in samples]),
            "inter_token_rank_mean": _summary(
                [sample["inter_token"]["mean_ms"] for sample in samples]
            ),
            "overlap_ms_per_decode_rank_mean": float(
                np.mean([sample["overlap_ms_per_decode"] for sample in samples])
            ),
            "max_abs_logit_error": max(sample["max_abs_logit_error"] for sample in samples),
            "rank0_timeline_sample": samples[0]["timeline_sample"],
        }
    sequential = result["modes"]["sequential"]["inter_token_rank_mean"]["mean_ms"]
    overlapped = result["modes"]["overlapped"]["inter_token_rank_mean"]["mean_ms"]
    result["measured_itl_change_percent"] = (overlapped / sequential - 1) * 100
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def finite_summary_check(result: dict) -> None:
    for mode in result["modes"].values():
        if not math.isfinite(mode["inter_token_rank_mean"]["mean_ms"]):
            raise ValueError("non-finite benchmark result")
