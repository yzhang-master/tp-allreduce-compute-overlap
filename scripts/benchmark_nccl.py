import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from tp_overlap import ModelConfig, TensorParallelLM, TinyCausalLM


def main() -> None:
    if not torch.cuda.is_available() or not dist.is_nccl_available():
        raise SystemExit("NCCL benchmark pending: CUDA and NCCL are required")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    try:
        config = ModelConfig(hidden_size=64, intermediate_size=128, num_heads=8, vocab_size=128)
        reference = TinyCausalLM(config).to(device)
        prompt = [2, 5, 11, 17, 23]
        result = {"backend": "nccl", "world_size": world_size, "modes": {}}
        for mode, asynchronous in (("sequential", False), ("overlapped", True)):
            latencies = []
            max_error = 0.0
            for repeat in range(6):
                model = TensorParallelLM(
                    reference, rank, world_size, output_chunks=2, asynchronous=asynchronous
                ).to(device)
                logits, cache = model.prefill(prompt)
                reference_logits, reference_cache = reference.prefill(prompt)
                for _ in range(8):
                    token = int(logits[-1].argmax())
                    torch.cuda.synchronize()
                    start = time.perf_counter_ns()
                    logits, cache = model.decode(token, cache)
                    torch.cuda.synchronize()
                    elapsed_ms = (time.perf_counter_ns() - start) / 1e6
                    reference_logits, reference_cache = reference.decode(token, reference_cache)
                    error = float((logits[-1] - reference_logits).abs().max())
                    max_error = max(max_error, error)
                    if repeat:
                        latencies.append(elapsed_ms)
            result["modes"][mode] = {
                "inter_token_mean_ms": float(np.mean(latencies)),
                "inter_token_p95_ms": float(np.percentile(latencies, 95)),
                "max_abs_logit_error": max_error,
            }
        if rank == 0:
            result["gpu"] = torch.cuda.get_device_name(device)
            result["torch"] = torch.__version__
            output = Path(f"results/nccl_tp{world_size}.json")
            output.parent.mkdir(exist_ok=True)
            output.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result, indent=2))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
