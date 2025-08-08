import time

import pytest
import torch

from tp_overlap.overlap import OverlapTimeline, row_parallel_gemm


class FakeWork:
    def __init__(self, on_wait) -> None:
        self.on_wait = on_wait

    def wait(self):
        self.on_wait()


def _run(monkeypatch, asynchronous: bool):
    monkeypatch.setattr("torch.distributed.is_initialized", lambda: True)
    calls = []
    waits = []

    def fake_all_reduce(tensor, async_op):
        calls.append((tensor.clone(), async_op))
        return FakeWork(lambda: waits.append(time.perf_counter_ns())) if async_op else None

    monkeypatch.setattr("torch.distributed.all_reduce", fake_all_reduce)
    torch.set_num_threads(1)
    local_input = torch.arange(48, dtype=torch.float32).reshape(8, 6) / 10
    weight = torch.linspace(-0.4, 0.5, 48, dtype=torch.float32).reshape(8, 6)
    timeline = OverlapTimeline()
    output = row_parallel_gemm(
        local_input,
        weight,
        output_chunks=2,
        asynchronous=asynchronous,
        timeline=timeline,
        layer="test",
    )
    return output, local_input @ weight.T, timeline, calls, waits


def test_async_work_is_outstanding_during_next_gemm(monkeypatch):
    output, expected, timeline, calls, waits = _run(monkeypatch, True)
    torch.testing.assert_close(output, expected)
    assert [call[1] for call in calls] == [True, True]
    assert timeline.overlap_seconds() > 0
    first_reduce = next(event for event in timeline.events if event.kind == "allreduce")
    second_gemm = next(
        event for event in timeline.events if event.kind == "gemm" and event.chunk == 1
    )
    assert first_reduce.start_ns < second_gemm.start_ns
    assert second_gemm.end_ns <= first_reduce.end_ns
    assert waits[0] >= second_gemm.end_ns


def test_synchronous_path_has_no_overlap(monkeypatch):
    output, expected, timeline, calls, waits = _run(monkeypatch, False)
    torch.testing.assert_close(output, expected)
    assert [call[1] for call in calls] == [False, False]
    assert not waits
    assert timeline.overlap_seconds() == 0


def test_invalid_chunking_is_rejected(monkeypatch):
    monkeypatch.setattr("torch.distributed.is_initialized", lambda: True)
    with pytest.raises(ValueError, match="divide"):
        row_parallel_gemm(
            torch.ones(2, 4),
            torch.ones(6, 4),
            output_chunks=4,
            asynchronous=True,
            timeline=OverlapTimeline(),
            layer="test",
        )
