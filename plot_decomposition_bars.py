# Fig-4-style decomposition bars: per-direction split and lean/push rotation.
# Reads phase0_out/summary.csv (main-experiment, non-variant rows), sorts by the
# paper's pooled metric (descending, Fig-4 order) so the lean panel is monotone
# by construction and the low-total end shows the shift-dominated cluster.
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).parent / "phase0_out"

C_ABOVE, C_BELOW = "#7b3294", "#e66101"   # purple / orange (PuOr, colorblind-safe)
C_LEAN, C_PUSH = "#1f77b4", "#d62728"     # match fig_note_ladder

rows = []
with open(OUT / "summary.csv") as f:
    for r in csv.DictReader(f):
        if r["is_variant"] == "True":
            continue
        g = lambda k: float(r[k])
        lean, lean_lo, lean_hi = g("pooled") / 2, g("pooled_lo") / 2, g("pooled_hi") / 2
        # delta = B_below - B_above, push = -delta/2
        push, push_lo, push_hi = -g("delta") / 2, -g("delta_hi") / 2, -g("delta_lo") / 2
        rows.append(dict(
            name=r["display"], tot=g("pooled"),
            ba=g("B_above"), ba_lo=g("B_above_lo"), ba_hi=g("B_above_hi"),
            bb=g("B_below"), bb_lo=g("B_below_lo"), bb_hi=g("B_below_hi"),
            lean=lean, lean_lo=lean_lo, lean_hi=lean_hi,
            push=push, push_lo=push_lo, push_hi=push_hi,
        ))

rows.sort(key=lambda r: r["tot"], reverse=True)   # top = most biased, Fig-4 order
ys = range(len(rows))
shift_dom = [abs(r["push"]) > r["lean"] for r in rows]


def draw_panel(ax, rows, k1, k2, c1, c2, lab1, lab2, show_names):
    h = 0.38
    for i, r in enumerate(rows):
        for k, c, off, lab in [(k1, c1, -h / 2, lab1), (k2, c2, h / 2, lab2)]:
            v, lo, hi = r[k], r[k + "_lo"], r[k + "_hi"]
            ax.barh(i + off, v, height=h, color=c, alpha=0.85,
                    label=lab if i == 0 else None)
            ax.plot([lo, hi], [i + off, i + off], color="0.25", lw=0.9, zorder=3)
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_ylim(len(rows) - 0.5, -0.5)   # most biased on top
    ax.set_yticks(list(ys))
    if show_names:
        ax.set_yticklabels(
            [r["name"] for r in rows],
            fontsize=8,
        )
        for tick, sd in zip(ax.get_yticklabels(), shift_dom):
            if sd:
                tick.set_fontweight("bold")
    else:
        ax.set_yticklabels([])
    ax.grid(axis="x", alpha=0.25, lw=0.5)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.005), ncol=2,
              fontsize=8, frameon=False, borderaxespad=0)


def panel_a(ax, show_names=True):
    draw_panel(ax, rows, "ba", "bb", C_ABOVE, C_BELOW,
               "bias_above (above-good arm)", "bias_below (below-good arm)",
               show_names)
    ax.set_xlabel("per-direction bias vs. no-bet baseline")
    ax.set_title("Per-direction split", fontsize=11, pad=24)


def panel_b(ax, show_names=False):
    draw_panel(ax, rows, "lean", "push", C_LEAN, C_PUSH,
               "lean = (above+below)/2", "push = (above−below)/2",
               show_names)
    ax.set_xlabel("lean / push")
    ax.set_title("Rotated coordinates", fontsize=11, pad=24)


# combined two-panel figure
fig, (axA, axB) = plt.subplots(
    1, 2, figsize=(11, 0.34 * len(rows) + 2.2), sharey=False,
    gridspec_kw={"width_ratios": [1.25, 1]})
panel_a(axA, show_names=True)
panel_b(axB, show_names=False)
axB.set_yticks(list(ys))
axB.set_yticklabels([])
fig.suptitle("Donation Bet main experiment: the paper's per-model bias, split "
             "(models sorted by the paper's pooled metric; bold = |push| > lean)",
             fontsize=10.5)
fig.tight_layout(rect=(0, 0, 1, 0.965))
fig.savefig(OUT / "fig_decomposition_bars.png", dpi=200)
plt.close(fig)

# standalone versions of each panel
for fn, painter in [("fig_bars_above_below.png", panel_a),
                    ("fig_bars_lean_push.png", panel_b)]:
    f, ax = plt.subplots(figsize=(7.5, 0.34 * len(rows) + 2))
    painter(ax, show_names=True)
    f.tight_layout()
    f.savefig(OUT / fn, dpi=200)
    plt.close(f)

print("wrote", OUT / "fig_decomposition_bars.png",
      "+ fig_bars_above_below.png + fig_bars_lean_push.png")
