from tp_overlap.benchmark import _summary


def test_latency_summary_uses_observed_samples():
    summary = _summary([1.0, 2.0, 3.0, 4.0])
    assert summary["mean_ms"] == 2.5
    assert summary["p50_ms"] == 2.5
    assert summary["p95_ms"] > summary["p50_ms"]
