# tp-allreduce-compute-overlap

> **Era:** 2025+ · Staff AI Engineer focus area (distributed LLM inference). Tensor-parallel allreduce/compute overlap techniques from 2022–2025, reimplemented in 2026 with Python 3.12 and PyTorch 2.x.

A real tensor-parallel decoder layer with head-sharded causal attention, sharded SwiGLU, incremental KV state, and a custom sliced allreduce schedule. Each row-parallel output projection is split into independent GEMMs; an asynchronous `torch.distributed.Work` remains outstanding while the next GEMM executes.

## Problem

Tensor parallelism splits GEMMs across ranks, but a row-parallel projection must sum every rank's partial output. A synchronous implementation waits for that reduction before starting the next operation. This repository overlaps independent output slices without changing the mathematical result:

```text
partial_h[r] = local_input @ local_weight[h].T
output_h     = Σ ranks partial_h[r]
```

The slices are disjoint, so reducing slice 0 cannot race with computation of slice 1.

## Method

### Actual tensor-parallel inference

The unsharded `TinyCausalLM` is the numerical reference. `TensorParallelLM` owns only local Q/K/V head rows, local attention-output input columns, local gate/up rows, and local down-projection input columns. Embeddings, norms, and the LM head are replicated. Attention KV caches are local to each rank. Prefill uses causal attention; decode appends one token to each rank's KV cache and generates the next token by argmax.

Both the attention output projection and MLP down projection use `row_parallel_gemm`:

```text
GEMM(slice 0) → launch allreduce 0 (async)
                GEMM(slice 1) → wait 0 → launch allreduce 1 (async)
                                  GEMM(slice 2) → wait 1 ...
```

Only one collective per rank is outstanding. `Work.wait()` precedes every dependent operation, and each allreduce mutates a tensor not used by the concurrent GEMM. The synchronous control performs the same GEMMs and reductions but waits immediately. No timeline is inferred from running sequential collectives.

### Timeline semantics

`OverlapTimeline` records host-side `perf_counter_ns` intervals for GEMMs and asynchronous `Work` lifetimes. `overlap_seconds` is the union of their intersections, avoiding double counting. On Gloo this measures actual asynchronous API/execution scheduling on CPU. It is not an NCCL kernel trace: CUDA kernel-level timelines would require CUDA events or a profiler.

## Results

Measured on 2026-11-14, Linux x86_64, Python 3.12.3, PyTorch 2.15.0+cpu, one torch thread per rank. Two real Gloo processes ran five measured repetitions after one warmup, with eight decode tokens per repetition. Values below are means of per-rank means; raw intervals and one rank-0 timeline are in `results/cpu_gloo.json`.

| TP / backend | Schedule | Mean prefill (ms) | Mean inter-token latency (ms) | Work/GEMM overlap per decode (ms) | Max abs logit error |
|---|---|---:|---:|---:|---:|
| TP=2, CPU Gloo | synchronous | 23.479 | 27.982 | 0.000 | 2.98e-7 |
| TP=2, CPU Gloo | asynchronous | 29.652 | 26.659 | 0.354 | 2.98e-7 |

The measured inter-token change is **−4.73%** (lower is better) in this run. This tiny, noisy CPU workload is not evidence of a production GPU speedup. The timeline directly demonstrates overlap even when end-to-end latency is dominated by host and process-group overhead.

NCCL inter-token latency is **pending** on this CPU-only machine. The harness performs the same sharded inference on CUDA and checks full-model logits; it does not substitute synthetic collectives.

| Backend | TP=2 | TP=4 | TP=8 |
|---|---|---|---|
| NCCL inter-token latency | pending | pending | pending |

Run each configuration on an eight-GPU host:

```bash
torchrun --standalone --nproc-per-node=2 scripts/benchmark_nccl.py
torchrun --standalone --nproc-per-node=4 scripts/benchmark_nccl.py
torchrun --standalone --nproc-per-node=8 scripts/benchmark_nccl.py
```

Each command writes `results/nccl_tp<N>.json` only after running on the requested ranks.

## Correctness

The nine tests cover incremental attention against full causal recomputation, sharding constraints, async scheduling, synchronous negative control, latency accounting, and real two-rank Gloo prefill/decode. Every rank compares four autoregressive decode steps with the full model and records the overlap timeline. The async test checks the actual wait-call timestamp, not just timeline bookkeeping. A mutation that inserted an immediate `Work.wait()` failed that assertion (pytest exit 1); the implementation was restored and all nine tests passed.

## Reproduce

Using the shared portfolio environment:

```bash
cd /home/pus/Documents/github/tp-allreduce-compute-overlap
/home/pus/.venvs/portfolio/bin/ruff check .
/home/pus/.venvs/portfolio/bin/pytest -q
PYTHONPATH=src /home/pus/.venvs/portfolio/bin/python scripts/smoke.py
PYTHONPATH=src /home/pus/.venvs/portfolio/bin/python scripts/benchmark.py
PYTHONPATH=src /home/pus/.venvs/portfolio/bin/python scripts/benchmark.py --world-size 4
```

For a fresh environment, install PyTorch and `pip install -e ".[dev]"`. The benchmark overwrites `results/cpu_gloo.json`; wall-clock results vary with machine load.

## Limitations

- The model is one small deterministic decoder layer, not a production multi-layer LLM or tokenizer.
- Gloo measurements include Python and process-group overhead; five repetitions are smoke-scale, not a controlled performance study.
- Output slicing increases collective count. Whether overlap wins depends on GEMM size, transport, and rank count.
- NCCL and TP=8 execution are unmeasured here; no GPU performance or scaling claim is made.
- The CPU timeline tracks `Work` outstanding intervals, not individual Gloo device threads or CUDA kernels.

## References

- Shoeybi et al. *Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism.* 2019. arXiv:1909.08053
- Narayanan et al. *Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM.* 2021. arXiv:2104.04473
- Kwon et al. *Efficient Memory Management for Large Language Model Serving with PagedAttention.* 2023. arXiv:2309.06180
- PyTorch documentation. [`torch.distributed.all_reduce(async_op=True)`](https://pytorch.org/docs/stable/distributed.html)

## License

MIT
