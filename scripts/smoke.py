from tp_overlap.benchmark import finite_summary_check, run_gloo_benchmark

if __name__ == "__main__":
    result = run_gloo_benchmark(world_size=2, decode_tokens=4, repeats=1)
    finite_summary_check(result)
    for mode, values in result["modes"].items():
        print(
            f"{mode}: ITL={values['inter_token_rank_mean']['mean_ms']:.3f} ms, "
            f"overlap={values['overlap_ms_per_decode_rank_mean']:.3f} ms/decode, "
            f"max_error={values['max_abs_logit_error']:.3g}"
        )
