"""Phase 0 Step 3: born-vs-made decomposition of the per-direction bias.

Using the paper's trajectory-judge cache (first in-CoT estimate per rollout),
decompose each direction's outcome bias on the trajectory-parsed subset:

    born_dir  = bias of the FIRST in-CoT estimate vs baseline firsts
    made_dir  = bias of the FINAL answer  vs baseline finals  -  born_dir

so  final-bias_dir = born_dir + made_dir, all in fraction-on-good-side units,
each referenced against the baseline's own first/final rates (paired subset).
Also reports the baseline's own outcome drift q_final - q_first (negative =
natural downward current) and trajectory parse rates per condition
(selection diagnostic).

Outputs: phase0_out/step3_born_made.csv, fig_born_made.png
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE / "ext" / "value_leakage"
OUT = HERE / "phase0_out"
sys.path.insert(0, str(REPO))

from donation_bet.trajectories.data import load_model_data  # noqa: E402
from shared.runner import CacheOnlyMiss  # noqa: E402
from shared.models import MODELS  # noqa: E402

EXPERIMENT = "main_experiment_accurate"
N_BOOT = 1000
BOOT_SEED = 0
VARIANT_SUFFIXES = ("-claudeai", "-conglomerate", "-dont-bias-prompt")
DATA = HERE / "ext" / "value_leakage_data" / "final_data"


def cell_arrays(tdf, pk, direction, t):
    """(first_good_above, final_good_above) bool arrays for one cell."""
    sub = tdf[(tdf.prompt_key == pk) & (tdf.direction == direction)
              & tdf.trajectory.notna()]
    first = sub.trajectory.map(lambda tr: tr[0]).to_numpy(dtype=float)
    final = sub.estimate.to_numpy(dtype=float)
    return first > t, final > t


def model_decomposition(tdf, thresholds, rng):
    """Equal-question decomposition + paired bootstrap CIs."""
    cells = {}  # (pk) -> dict(direction -> (first_above, final_above))
    for pk, t in thresholds.items():
        by_dir = {}
        for d in ("baseline", "above_good", "below_good"):
            fa, fi = cell_arrays(tdf, pk, d, t)
            if len(fa) == 0:
                by_dir = None
                break
            by_dir[d] = (fa, fi)
        if by_dir:
            cells[pk] = by_dir

    if not cells:
        return None

    def stats_from(draw):
        """draw(cell) -> (first_above_mean, final_above_mean) per direction."""
        agg = dict(born_above=0.0, made_above=0.0, born_below=0.0,
                   made_below=0.0, base_drift=0.0)
        nq = len(cells)
        for by_dir in cells.values():
            (bqf, bqF) = draw(by_dir["baseline"])
            (aqf, aqF) = draw(by_dir["above_good"])
            (lqf, lqF) = draw(by_dir["below_good"])
            born_a = aqf - bqf
            final_a = aqF - bqF
            born_b = (1 - lqf) - (1 - bqf)
            final_b = (1 - lqF) - (1 - bqF)
            agg["born_above"] += born_a / nq
            agg["made_above"] += (final_a - born_a) / nq
            agg["born_below"] += born_b / nq
            agg["made_below"] += (final_b - born_b) / nq
            agg["base_drift"] += (bqF - bqf) / nq
        return agg

    def point_draw(cell):
        fa, fi = cell
        return fa.mean(), fi.mean()

    point = stats_from(point_draw)

    boots = {k: np.zeros(N_BOOT) for k in point}
    for i in range(N_BOOT):
        def bdraw(cell):
            fa, fi = cell
            idx = rng.integers(0, len(fa), len(fa))
            return fa[idx].mean(), fi[idx].mean()
        s = stats_from(bdraw)
        for k in boots:
            boots[k][i] = s[k]
    cis = {k: (float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975)))
           for k, v in boots.items()}
    return point, cis, len(cells)


def main():
    OUT.mkdir(exist_ok=True)
    cache_models = sorted(
        d.name for d in (DATA / "cache").iterdir()
        if d.is_dir() and d.name in MODELS
        and not d.name.endswith(VARIANT_SUFFIXES)
    )
    rows, failures = [], []
    for key in cache_models:
        try:
            df, tdf, display = load_model_data(
                key, experiment=EXPERIMENT, cache_only=True)
        except CacheOnlyMiss as e:
            failures.append((key, str(e)[:100]))
            continue
        thr = (df.dropna(subset=["threshold"])
                 .groupby("prompt_key").threshold.first().to_dict())
        # parse-rate diagnostics (selection check)
        pr = {}
        for d in ("baseline", "above_good", "below_good"):
            sub = tdf[tdf.direction == d]
            pr[d] = float(sub.trajectory.notna().mean()) if len(sub) else np.nan
        rng = np.random.default_rng(BOOT_SEED)
        out = model_decomposition(tdf, thr, rng)
        if out is None:
            failures.append((key, "no usable cells"))
            continue
        point, cis, nq = out
        row = dict(model_key=key, display=display, n_questions=nq,
                   parse_base=pr["baseline"], parse_above=pr["above_good"],
                   parse_below=pr["below_good"])
        for k, v in point.items():
            row[k] = v
            row[f"{k}_lo"], row[f"{k}_hi"] = cis[k]
        row["final_above"] = point["born_above"] + point["made_above"]
        row["final_below"] = point["born_below"] + point["made_below"]
        rows.append(row)
        print(f"{display:26s} above: born {point['born_above']:+.3f} + made {point['made_above']:+.3f}"
              f" | below: born {point['born_below']:+.3f} + made {point['made_below']:+.3f}"
              f" | base drift {point['base_drift']:+.3f}"
              f" | parse {pr['baseline']:.0%}/{pr['above_good']:.0%}/{pr['below_good']:.0%}")

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "step3_born_made.csv", index=False)
    if failures:
        print("\nSKIPPED (cache miss):")
        for k, e in failures:
            print(f"  {k}: {e}")

    # order by Phase-0 delta for visual continuity
    s0 = pd.read_csv(OUT / "summary.csv")
    order = s0[~s0.is_variant].sort_values("delta").display.tolist()
    res["o"] = res.display.map({d: i for i, d in enumerate(order)})
    res = res.sort_values("o")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y = np.arange(len(res))
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.34 * len(res) + 2),
                             sharey=True)
    for ax, side in zip(axes, ("above", "below")):
        born = res[f"born_{side}"].to_numpy()
        made = res[f"made_{side}"].to_numpy()
        ax.barh(y, born, color="#8c6bb1", label="born (first estimate)")
        ax.barh(y, made, left=born, color="#fdb863", label="made (revisions)")
        tot = born + made
        ax.scatter(tot, y, color="k", s=12, zorder=3, label="final bias")
        ax.axvline(0, color="gray", lw=0.7)
        ax.set_title(f"{side}_good: bias toward good side")
        ax.set_xlabel("fraction-on-good-side bias vs baseline")
    axes[0].set_yticks(y, res.display, fontsize=7)
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Born vs made: where each direction's bias enters "
                 "(paper's trajectory cache, paired subset)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_born_made.png", dpi=200)
    print(f"\nWrote {OUT}/step3_born_made.csv, fig_born_made.png")


if __name__ == "__main__":
    main()
