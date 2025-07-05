import time
from dataclasses import asdict, dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class TimelineEvent:
    layer: str
    chunk: int
    kind: str
    start_ns: int
    end_ns: int


class OverlapTimeline:
    def __init__(self) -> None:
        self.events: list[TimelineEvent] = []

    def add(self, layer: str, chunk: int, kind: str, start_ns: int, end_ns: int) -> None:
        self.events.append(TimelineEvent(layer, chunk, kind, start_ns, end_ns))

    def overlap_seconds(self) -> float:
        total = 0.0
        for compute in (event for event in self.events if event.kind == "gemm"):
            intervals = []
            for communication in (event for event in self.events if event.kind == "allreduce"):
                start = max(compute.start_ns, communication.start_ns)
                end = min(compute.end_ns, communication.end_ns)
                if end > start:
                    intervals.append((start, end))
            for start, end in _merge_intervals(intervals):
                total += end - start
        return total / 1e9

    def as_dicts(self) -> list[dict[str, int | str]]:
        return [asdict(event) for event in self.events]


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def row_parallel_gemm(
    local_input: torch.Tensor,
    weight: torch.Tensor,
    *,
    output_chunks: int,
    asynchronous: bool,
    timeline: OverlapTimeline,
    layer: str,
) -> torch.Tensor:
    """Reduce disjoint output slices, computing slice i+1 while slice i is in flight."""
    if local_input.shape[-1] != weight.shape[1]:
        raise ValueError("input and sharded weight dimensions differ")
    hidden = weight.shape[0]
    if output_chunks <= 0 or hidden % output_chunks:
        raise ValueError("output_chunks must be positive and divide the output dimension")
    if not dist.is_initialized():
        raise RuntimeError("an initialized process group is required")

    width = hidden // output_chunks
    outputs: list[torch.Tensor] = []
    pending: tuple[dist.Work, int, int] | None = None
    for chunk in range(output_chunks):
        begin = chunk * width
        end = begin + width
        compute_start = time.perf_counter_ns()
        partial = local_input @ weight[begin:end].T
        compute_end = time.perf_counter_ns()
        timeline.add(layer, chunk, "gemm", compute_start, compute_end)

        if pending is not None:
            previous_work, previous_chunk, communication_start = pending
            previous_work.wait()
            timeline.add(
                layer, previous_chunk, "allreduce", communication_start, time.perf_counter_ns()
            )

        output = partial.contiguous()
        communication_start = time.perf_counter_ns()
        work = dist.all_reduce(output, async_op=asynchronous)
        if asynchronous:
            if work is None:
                raise RuntimeError("asynchronous allreduce did not return a Work handle")
            pending = (work, chunk, communication_start)
        else:
            timeline.add(layer, chunk, "allreduce", communication_start, time.perf_counter_ns())
        outputs.append(output)

    if pending is not None:
        work, chunk, communication_start = pending
        work.wait()
        timeline.add(layer, chunk, "allreduce", communication_start, time.perf_counter_ns())
    return torch.cat(outputs, dim=-1)
