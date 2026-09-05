"""Compare fresh OpenRouter conversations against the paper's cached results.

Loads the fresh caches produced by fresh_batch_run.py (cache-only — never
samples), computes the same per-direction statistics as Phase 0 (bias_above /
bias_below vs the FRESH baseline split q, lean, push, pooled, bootstrap CIs),
and prints them side by side with the Phase 0 numbers from the paper's data
(phase0_out/summary.csv).

Usage:
    python3 fresh_batch_analyze.py                     # full batch comparison
    python3 fresh_batch_analyze.py --experiment smoke_accurate
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# phase0 module gives us question_stats/bootstrap_model (pure functions);
# fresh_batch_run (imported after) repoints the runner caches to fresh_batch/
# and registers the or-* models.
import phase0_unpooled_bias as p0  # noqa: E402
import fresh_batch_run as fb  # noqa: E402
from shared.get_main_dfs import get_main_dfs  # noqa: E402
import shared.runner as runner  # noqa: E402

runner.CACHE_DIR = str(fb.FRESH / "cache")
runner.ESTIMATE_JUDGE_CACHE_ROOT = str(fb.FRESH / "estimate_judge_cache")

FRESH_TO_CACHED = {
    "or-qwen3.5-35": "Qwen3.5-35B-A3B",
    "or-qwen3.6-35": "Qwen3.6-35B-A3B",
    "or-gpt-5.5-medium": "gpt-5.5-medium",
}


def fmt(x, lo=None, hi=None):
    if lo is None:
        return f"{x:+.3f}"
    return f"{x:+.3f} [{lo:+.3f},{hi:+.3f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="main_experiment_accurate",
                    choices=["main_experiment_accurate", "smoke_accurate"])
    args = ap.parse_args()

    cached = pd.read_csv(HERE / "phase0_out" / "summary.csv")
    cached = cached[~cached.is_variant].set_index("display")

    rows = []
    per_q_rows = []
    for key, cached_name in FRESH_TO_CACHED.items():
        try:
            df, thresholds, display = get_main_dfs(
                args.experiment, [key], cache_only=True)[key]
        except Exception as e:
            print(f"{key}: no fresh data yet ({type(e).__name__})")
            continue
        qrows = p0.question_stats(df, thresholds)
        if not qrows:
            print(f"{key}: fresh data present but no usable questions")
            continue
        qdf = pd.DataFrame(qrows)
        cis = p0.bootstrap_model(qrows)
        B_above, B_below = qdf.bias_above.mean(), qdf.bias_below.mean()
        rows.append(dict(
            key=key, cached_name=cached_name, n_questions=len(qrows),
            n_rows=len(df),
            B_above=B_above, B_above_ci=cis["B_above_ci"],
            B_below=B_below, B_below_ci=cis["B_below_ci"],
            lean=(B_above + B_below) / 2, push=(B_above - B_below) / 2,
            pooled=qdf.pooled.mean(), pooled_ci=cis["pooled_ci"],
            delta_ci=cis["delta_ci"],
        ))
        for r in qrows:
            per_q_rows.append(dict(model=cached_name, **r))

    if not rows:
        print("No fresh data found for any model.")
        return

    print(f"\n=== Fresh (OpenRouter, {args.experiment}) vs cached (paper) ===")
    for r in rows:
        c = cached.loc[r["cached_name"]]
        c_lean = (c.B_above + c.B_below) / 2
        c_push = (c.B_above - c.B_below) / 2
        print(f"\n{r['cached_name']}  (fresh rows: {r['n_rows']}, "
              f"questions: {r['n_questions']})")
        print(f"  {'':12s} {'fresh':>28s} {'cached (paper data)':>22s}")
        print(f"  {'bias_above':12s} {fmt(r['B_above'], *r['B_above_ci']):>28s} "
              f"{fmt(c.B_above, c.B_above_lo, c.B_above_hi):>22s}")
        print(f"  {'bias_below':12s} {fmt(r['B_below'], *r['B_below_ci']):>28s} "
              f"{fmt(c.B_below, c.B_below_lo, c.B_below_hi):>22s}")
        print(f"  {'lean':12s} {fmt(r['lean']):>28s} {fmt(c_lean):>22s}")
        print(f"  {'push':12s} {fmt(r['push']):>28s} {fmt(c_push):>22s}")
        print(f"  {'pooled':12s} {fmt(r['pooled'], *r['pooled_ci']):>28s} "
              f"{fmt(c.pooled, c.pooled_lo, c.pooled_hi):>22s}")
        gap_lo, gap_hi = r["delta_ci"]
        same_sign_push = np.sign(r["push"]) == np.sign(c_push)
        sig = (gap_lo > 0 or gap_hi < 0)
        print(f"  push sign matches cached: {same_sign_push} | "
              f"fresh direction-gap significant: {sig}")

    if per_q_rows and args.experiment == "main_experiment_accurate":
        pq = pd.DataFrame(per_q_rows)
        out = HERE / "phase0_out" / "fresh_per_question.csv"
        pq.to_csv(out, index=False)
        print(f"\nPer-question fresh stats written to {out}")


if __name__ == "__main__":
    main()
