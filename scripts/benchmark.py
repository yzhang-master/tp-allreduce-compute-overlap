import argparse
import json
from pathlib import Path

from tp_overlap.benchmark import finite_summary_check, run_gloo_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark real multi-process Gloo TP inference")
    parser.add_argument("--world-size", type=int, default=2, choices=(2, 4))
    parser.add_argument("--decode-tokens", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("results/cpu_gloo.json"))
    args = parser.parse_args()
    result = run_gloo_benchmark(
        world_size=args.world_size,
        decode_tokens=args.decode_tokens,
        repeats=args.repeats,
        output=args.output,
    )
    finite_summary_check(result)
    summaries = {
        mode: values["inter_token_rank_mean"] for mode, values in result["modes"].items()
    }
    print(json.dumps(summaries, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
