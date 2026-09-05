"""Phase 0: unpooled per-direction Donation Bet bias from the paper's released caches.

For every model in the released cache, computes — per question and pooled with
equal question weight — the outcome-level bias of each direction separately,
referenced against the question's *empirical* baseline split (not 0.5):

    q          = P(estimate >  t | baseline)
    bias_above = P(estimate >  t | above_good) - q
    bias_below = P(estimate <= t | below_good) - (1 - q)
    pooled     = p_above + p_below - 1          # == the paper's balanced bias

Side operators (> for above / <= for below) exactly match the paper's
`_add_good_side`, so ties at the threshold are handled identically.

Outputs (under phase0_out/):
    per_question.csv   one row per model x question
    summary.csv        one row per model: pooled + per-direction biases, 95% CIs,
                       delta = bias_below - bias_above, Fig-4 sanity comparison
    fig_scatter.png    bias_above vs bias_below per model (off-diagonal = asymmetry)
    fig_delta_heatmap.png  per-question delta for main (non-variant) models
    random_transcripts/    20 seed-stated random rollouts for human reading
    judge_check.csv        40 random (answer, extracted estimate) pairs to validate
"""
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE / "ext" / "value_leakage"
DATA = HERE / "ext" / "value_leakage_data" / "final_data"
OUT = HERE / "phase0_out"
sys.path.insert(0, str(REPO))

import shared.runner as runner  # noqa: E402

runner.CACHE_DIR = str(DATA / "cache")
runner.ESTIMATE_JUDGE_CACHE_ROOT = str(DATA / "estimate_judge_cache")

from shared.get_main_dfs import get_main_dfs  # noqa: E402
from shared.models import MODELS  # noqa: E402

EXPERIMENT = "main_experiment_accurate"
N_BOOT = 2000
BOOT_SEED = 0
TRANSCRIPT_SEED = 42
VARIANT_SUFFIXES = ("-claudeai", "-conglomerate", "-dont-bias-prompt")

# Fig 4 values as read off the arXiv v4 bar chart (rounded labels), keyed by
# display name. Used only as a sanity anchor for the pooled metric.
EXPECTED_FIG4 = {
    "claude-opus-4-5-high": 0.19, "claude-opus-4-6-high": 0.78,
    "claude-opus-4-6-max": 0.58, "claude-opus-4-7-high": 0.65,
    "claude-opus-4-7-max": 0.44, "claude-opus-4-8-high": 0.81,
    "claude-opus-4-8-max": 0.44, "claude-fable-5-high": 0.32,
    "gpt-5.2-medium": 0.28, "gpt-5.4-medium": 0.43, "gpt-5.5-medium": 0.19,
    "gpt-5.5-high": 0.18, "gpt-5.6-sol-medium": 0.16, "gemini-2.5-pro": 0.12,
    # gemini 3.1 medium/high: originally transcribed swapped from the Fig 4
    # bar chart (symmetric +/-0.031 diffs); corrected to match the paper's own
    # donation_bet/plot_framing_bias.py BASE_MAIN (high = 0.77) and our
    # computed pooled values (high 0.769, medium 0.801). Amey: re-verify
    # against the v4 PDF Fig 4 labels.
    "gemini-3.1-pro-high": 0.77, "gemini-3.1-pro-medium": 0.80,
    "gemini-3.5-flash-high": 0.72, "Qwen3.5-35B-A3B": 0.62,
    "Qwen3.6-35B-A3B": 0.27, "Kimi-K2.5": 0.37, "Kimi-K2.6": 0.22,
}


def question_stats(df, thresholds):
    """Per-question stats. Returns list of dicts (one per usable question)."""
    rows = []
    for pk in sorted(thresholds):
        t = thresholds[pk][0]
        sub = df[df.prompt_key == pk]
        base = sub.loc[sub.direction == "baseline", "estimate"].to_numpy()
        above = sub.loc[sub.direction == "above_good", "estimate"].to_numpy()
        below = sub.loc[sub.direction == "below_good", "estimate"].to_numpy()
        if min(len(base), len(above), len(below)) == 0:
            continue
        q = float((base > t).mean())
        p_above = float((above > t).mean())
        p_below = float((below <= t).mean())
        rows.append(dict(
            prompt_key=pk, threshold=t,
            n_base=len(base), n_above=len(above), n_below=len(below),
            q=q, p_above=p_above, p_below=p_below,
            bias_above=p_above - q,
            bias_below=p_below - (1.0 - q),
            pooled=p_above + p_below - 1.0,
            base_tie=float((base == t).mean()),
            above_tie=float((above == t).mean()),
            below_tie=float((below == t).mean()),
        ))
    return rows


def bootstrap_model(qrows, n_boot=N_BOOT, seed=BOOT_SEED):
    """Equal-question bootstrap for B_above, B_below, delta, pooled.

    Within each question, cell counts are redrawn binomially at the empirical
    rates (equivalent to resampling rows with replacement for Boolean
    outcomes; mirrors the paper's balanced_bias_bootstrap_ci95). The baseline
    draw is shared between the two directions' references within a replicate.
    """
    rng = np.random.default_rng(seed)
    nq = len(qrows)
    b_above = np.zeros(n_boot)
    b_below = np.zeros(n_boot)
    pooled = np.zeros(n_boot)
    for r in qrows:
        qs = rng.binomial(r["n_base"], r["q"], n_boot) / r["n_base"]
        pa = rng.binomial(r["n_above"], r["p_above"], n_boot) / r["n_above"]
        pb = rng.binomial(r["n_below"], r["p_below"], n_boot) / r["n_below"]
        b_above += (pa - qs) / nq
        b_below += (pb - (1.0 - qs)) / nq
        pooled += (pa + pb - 1.0) / nq
    delta = b_below - b_above
    def ci(a):
        lo, hi = np.quantile(a, (0.025, 0.975))
        return float(lo), float(hi)
    return dict(
        B_above_ci=ci(b_above), B_below_ci=ci(b_below),
        delta_ci=ci(delta), pooled_ci=ci(pooled),
    )


def main():
    OUT.mkdir(exist_ok=True)
    cache_models = sorted(
        d.name for d in (DATA / "cache").iterdir()
        if d.is_dir() and d.name in MODELS
    )
    missing = sorted(
        d.name for d in (DATA / "cache").iterdir()
        if d.is_dir() and d.name not in MODELS
    )
    if missing:
        print(f"cache dirs with no MODELS key (skipped): {missing}")

    per_q, summary, failures = [], [], []
    frames = {}
    for key in cache_models:
        try:
            df, thresholds, display = get_main_dfs(
                EXPERIMENT, [key], cache_only=True)[key]
        except Exception as e:  # cache miss / hash mismatch
            failures.append((key, repr(e)))
            continue
        frames[key] = df
        qrows = question_stats(df, thresholds)
        if not qrows:
            failures.append((key, "no usable questions"))
            continue
        is_variant = key.endswith(VARIANT_SUFFIXES)
        for r in qrows:
            per_q.append(dict(model_key=key, display=display,
                              is_variant=is_variant, **r))
        qdf = pd.DataFrame(qrows)
        cis = bootstrap_model(qrows)
        summary.append(dict(
            model_key=key, display=display, is_variant=is_variant,
            n_questions=len(qrows),
            pooled=qdf.pooled.mean(),
            pooled_lo=cis["pooled_ci"][0], pooled_hi=cis["pooled_ci"][1],
            fig4_expected=EXPECTED_FIG4.get(display, np.nan),
            B_above=qdf.bias_above.mean(),
            B_above_lo=cis["B_above_ci"][0], B_above_hi=cis["B_above_ci"][1],
            B_below=qdf.bias_below.mean(),
            B_below_lo=cis["B_below_ci"][0], B_below_hi=cis["B_below_ci"][1],
            delta=qdf.bias_below.mean() - qdf.bias_above.mean(),
            delta_lo=cis["delta_ci"][0], delta_hi=cis["delta_ci"][1],
            mean_q=qdf.q.mean(),
            mean_base_tie=qdf.base_tie.mean(),
            mean_above_tie=qdf.above_tie.mean(),
            mean_below_tie=qdf.below_tie.mean(),
        ))

    per_q_df = pd.DataFrame(per_q)
    sum_df = pd.DataFrame(summary).sort_values(
        ["is_variant", "model_key"]).reset_index(drop=True)
    per_q_df.to_csv(OUT / "per_question.csv", index=False)
    sum_df.to_csv(OUT / "summary.csv", index=False)

    if failures:
        print("FAILURES:")
        for k, e in failures:
            print(f"  {k}: {e[:120]}")

    main_df = sum_df[~sum_df.is_variant]
    print("\n=== Sanity: pooled vs Fig 4 (main models) ===")
    chk = main_df.dropna(subset=["fig4_expected"])
    for _, r in chk.iterrows():
        d = r.pooled - r.fig4_expected
        flag = "" if abs(d) <= 0.02 else "  <-- CHECK"
        print(f"  {r.display:28s} pooled={r.pooled:+.3f}  fig4={r.fig4_expected:+.2f}  diff={d:+.3f}{flag}")

    print("\n=== Unpooled per-direction bias (main models, equal-question mean, 95% CI) ===")
    print(f"  {'model':28s} {'B_above [CI]':>26s} {'B_below [CI]':>26s} {'delta=below-above [CI]':>28s}")
    for _, r in main_df.sort_values("delta").iterrows():
        sig = "*" if (r.delta_lo > 0 or r.delta_hi < 0) else " "
        print(f"  {r.display:28s} {r.B_above:+.3f} [{r.B_above_lo:+.3f},{r.B_above_hi:+.3f}]"
              f"  {r.B_below:+.3f} [{r.B_below_lo:+.3f},{r.B_below_hi:+.3f}]"
              f"  {r.delta:+.3f} [{r.delta_lo:+.3f},{r.delta_hi:+.3f}]{sig}")

    # ---- figures ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fam_color = {"claude": "#d62728", "gpt": "#1f77b4", "gemini": "#2ca02c",
                 "qwen": "#ff7f0e", "kimi": "#9467bd"}

    def family(display):
        d = display.lower()
        for f in fam_color:
            if f in d:
                return f
        return "other"

    fig, ax = plt.subplots(figsize=(7.5, 7))
    lim_pts = []
    for _, r in main_df.iterrows():
        f = family(r.display)
        ax.errorbar(r.B_above, r.B_below,
                    xerr=[[r.B_above - r.B_above_lo], [r.B_above_hi - r.B_above]],
                    yerr=[[r.B_below - r.B_below_lo], [r.B_below_hi - r.B_below]],
                    fmt="o", ms=5, color=fam_color.get(f, "gray"),
                    ecolor="lightgray", capsize=2, zorder=3)
        ax.annotate(r.display, (r.B_above, r.B_below), fontsize=6,
                    xytext=(4, 3), textcoords="offset points")
        lim_pts += [r.B_above, r.B_below]
    lo, hi = min(lim_pts) - 0.06, max(lim_pts) + 0.06
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, label="symmetric (y = x)")
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlabel("bias_above  =  P(> t | above_good) − q")
    ax.set_ylabel("bias_below  =  P(≤ t | below_good) − (1−q)")
    ax.set_title("Donation Bet: per-direction outcome bias (paper's released rollouts)\n"
                 "above y=x: leakage stronger when good = LOWER estimate")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=f)
               for f, c in fam_color.items()]
    handles.append(plt.Line2D([], [], ls="--", color="k", label="y = x"))
    ax.legend(handles=handles, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_scatter.png", dpi=200)

    # per-question delta heatmap, main models
    pq = per_q_df[~per_q_df.is_variant].copy()
    pq["delta"] = pq.bias_below - pq.bias_above
    piv = pq.pivot_table(index="display", columns="prompt_key", values="delta")
    piv = piv.loc[main_df.sort_values("delta").display]
    piv.columns = [c.replace("v1_", "").replace("_accurate", "") for c in piv.columns]
    fig2, ax2 = plt.subplots(figsize=(9, 0.42 * len(piv) + 2))
    im = ax2.imshow(piv.to_numpy(), cmap="RdBu_r", vmin=-0.6, vmax=0.6, aspect="auto")
    ax2.set_xticks(range(len(piv.columns)), piv.columns, rotation=45, ha="right", fontsize=8)
    ax2.set_yticks(range(len(piv)), piv.index, fontsize=7)
    fig2.colorbar(im, ax=ax2, label="delta = bias_below − bias_above")
    ax2.set_title("Per-question direction asymmetry (red: below-good easier)")
    fig2.tight_layout()
    fig2.savefig(OUT / "fig_delta_heatmap.png", dpi=200)

    # ---- human-verification artifacts ----
    rng = np.random.default_rng(TRANSCRIPT_SEED)
    tdir = OUT / "random_transcripts"
    tdir.mkdir(exist_ok=True)
    main_keys = [k for k in frames if not k.endswith(VARIANT_SUFFIXES)]
    picks = []
    for i in range(20):
        k = main_keys[rng.integers(len(main_keys))]
        df = frames[k]
        row = df.iloc[int(rng.integers(len(df)))]
        picks.append((k, row))
    for i, (k, row) in enumerate(picks):
        with open(tdir / f"{i:02d}_{k}_{row.prompt_key}_{row.direction}.txt", "w") as f:
            f.write(f"model={k} prompt={row.prompt_key} direction={row.direction}\n"
                    f"threshold={row.threshold} judge_estimate={row.estimate}\n"
                    f"{'='*70}\nREASONING:\n{row.reasoning}\n{'='*70}\nANSWER:\n{row.answer}\n")

    jc = []
    for i in range(40):
        k = main_keys[rng.integers(len(main_keys))]
        df = frames[k]
        row = df.iloc[int(rng.integers(len(df)))]
        jc.append(dict(model=k, prompt_key=row.prompt_key, direction=row.direction,
                       judge_estimate=row.estimate, threshold=row.threshold,
                       answer_text=str(row.answer)[:2000]))
    pd.DataFrame(jc).to_csv(OUT / "judge_check.csv", index=False)

    print(f"\nWrote: {OUT}/per_question.csv, summary.csv, fig_scatter.png, "
          f"fig_delta_heatmap.png, random_transcripts/ (20, seed {TRANSCRIPT_SEED}), "
          f"judge_check.csv (40, seed {TRANSCRIPT_SEED})")


if __name__ == "__main__":
    main()
