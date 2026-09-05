"""One-model illustration (gpt-5.5-high): what lean and push look like in the raw answers.

Two panels of ECDFs of the relative answer position (estimate - threshold)/threshold,
pooled over the 9 questions (x = 0 is the quoted threshold; curve height at x = 0
is the fraction of answers at or below the threshold).

  Left:  the paper's MAIN bet (charity vs nothing, n=100/cell, accurate frame):
         baseline vs high-is-good vs low-is-good.
  Right: the five charity-PAIR bets pooled (donation on BOTH sides, n=10/cell
         x 9 questions x 5 pairs = 450/direction, plain frame): baseline vs the
         two mirrored placements.

Output: phase0_out/fig_gpt55_story.png
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE / "ext" / "value_leakage"
DATA = HERE / "ext" / "value_leakage_data" / "final_data"
OUT = HERE / "phase0_out"
sys.path.insert(0, str(REPO))

import shared.runner as runner  # noqa: E402

runner.CACHE_DIR = str(DATA / "cache")
runner.ESTIMATE_JUDGE_CACHE_ROOT = str(DATA / "estimate_judge_cache")

from shared.get_main_dfs import get_main_dfs  # noqa: E402

MODEL = "gpt-5.5-high"
PAIR_EXPS = ["main_experiment_small_f5_dwb_greenpeace",
             "main_experiment_small_f5_redcross_unicef",
             "main_experiment_small_f5_mnm_amf",
             "main_experiment_small_f5_feeding_good360",
             "main_experiment_small_f5_wwf_tnc"]


def rel_disp(exp_names):
    """Pooled relative displacements per direction for a list of experiments.

    Baselines are deduplicated across experiments that share them (same
    values), so the shared plain baseline is counted once.
    """
    d = {"baseline": [], "above_good": [], "below_good": []}
    seen_baseline_keys = set()
    for exp in exp_names:
        df, thresholds, _ = get_main_dfs(exp, [MODEL], cache_only=True,
                                         raw=True)[MODEL]
        for pk in sorted(thresholds):
            t = thresholds[pk][0]
            sub = df[df.prompt_key == pk]
            for direction in d:
                if direction == "baseline":
                    key = (round(float(t)),
                           tuple(sub[sub.direction == "baseline"]
                                 .estimate.dropna().astype(float)))
                    if key in seen_baseline_keys:
                        continue
                    seen_baseline_keys.add(key)
                est = (sub[sub.direction == direction]
                       .estimate.dropna().astype(float).to_numpy())
                d[direction].append((est - t) / t)
    return {k: np.concatenate(v) for k, v in d.items()}


def ecdf(ax, arr, **kw):
    x = np.sort(arr)
    y = np.arange(1, len(x) + 1) / len(x)
    ax.step(x, y, where="post", **kw)


def main():
    main_d = rel_disp(["main_experiment_accurate"])
    pair_d = rel_disp(PAIR_EXPS)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    ax = axes[0]
    ecdf(ax, main_d["baseline"], color="gray", lw=2, label="no note (baseline)")
    ecdf(ax, main_d["above_good"], color="#1f77b4", lw=2,
         label="HIGH answer donates")
    ecdf(ax, main_d["below_good"], color="#d62728", lw=2,
         label="LOW answer donates")
    ax.set_title("The paper's bet: charity vs nothing\n(100 answers/condition"
                 " per question)")

    ax = axes[1]
    ecdf(ax, pair_d["baseline"], color="gray", lw=2, label="no note (baseline)")
    ecdf(ax, pair_d["above_good"], color="#1f77b4", lw=2,
         label="charity X if HIGH, charity Y if LOW")
    ecdf(ax, pair_d["below_good"], color="#d62728", lw=2,
         label="charity Y if HIGH, charity X if LOW")
    ax.set_title("Referee bets: a charity on BOTH sides\n(5 charity pairs"
                 " pooled, 450 answers/condition)")

    for ax in axes:
        ax.axvline(0, color="k", lw=1, ls="--")
        ax.text(0.02, 0.03, "quoted\nthreshold", fontsize=8, ha="left")
        ax.set_xlim(-1.05, 1.6)
        ax.set_xlabel("answer relative to the threshold  (answer − τ)/τ")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="lower right")
    axes[0].set_ylabel("fraction of answers at or below this value")

    # annotate the crossing heights at tau (fraction at-or-below threshold)
    for ax, dd in zip(axes, [main_d, pair_d]):
        for arr, color in [(dd["baseline"], "gray"),
                           (dd["above_good"], "#1f77b4"),
                           (dd["below_good"], "#d62728")]:
            frac = float((arr <= 0).mean())
            ax.plot([0], [frac], "o", ms=5, color=color)
            ax.annotate(f"{frac:.0%}", (0, frac), xytext=(-34, -2),
                        textcoords="offset points", fontsize=8, color=color)

    fig.suptitle("gpt-5.5-high: swap the charities, nothing changes — "
                 "the note itself pushes answers above the quoted number",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig_gpt55_story.png", dpi=200)
    print("wrote", OUT / "fig_gpt55_story.png")
    for name, dd in [("MAIN", main_d), ("PAIRS", pair_d)]:
        print(f"{name}: at-or-below tau: base={float((dd['baseline']<=0).mean()):.0%} "
              f"A={float((dd['above_good']<=0).mean()):.0%} "
              f"B={float((dd['below_good']<=0).mean()):.0%}  "
              f"exactly-at-tau: base={float((dd['baseline']==0).mean()):.1%} "
              f"A={float((dd['above_good']==0).mean()):.1%} "
              f"B={float((dd['below_good']==0).mean()):.1%}")


if __name__ == "__main__":
    main()
