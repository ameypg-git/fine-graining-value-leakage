# Clean-label version of fig_scatter.png (bias_above vs bias_below, 26 main
# models, y=x line). Same data/CIs as the original (summary.csv); only label
# placement fixed for the known collisions (gpt-5.5 pair, center cluster) and
# the in-image title dropped (the LaTeX caption carries the interpretation).
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).parent / "phase0_out"

FAMILY_COLORS = [("claude", "#d62728"), ("gpt", "#1f77b4"),
                 ("gemini", "#2ca02c"), ("qwen", "#ff7f0e"), ("kimi", "#9467bd")]


def family(name):
    n = name.lower()
    for key, c in FAMILY_COLORS:
        if key in n:
            return key, c
    return "other", "0.4"


# (dx_pt, dy_pt, ha) per display name; default is right of the point.
OFFSETS = {
    "gpt-5.5-medium": (-7, 4, "right"),
    "gpt-5.5-high": (7, -10, "left"),
    "gpt-5.2-medium": (-7, -2, "right"),
    "claude-fable-5-high": (7, -10, "left"),
    "Kimi-K2.5": (7, -11, "left"),
    "claude-opus-4-1": (-7, -2, "right"),
    "claude-sonnet-4-6-high": (2, 9, "left"),
    "gemini-3.1-pro-high": (-7, 2, "right"),
    "gemini-3.1-pro-medium": (7, -6, "left"),
    "gemini-3.5-flash-high": (-7, -3, "right"),
    "Qwen3.5-35B-A3B": (-7, -14, "right"),
    "claude-opus-4-6-high": (7, 5, "left"),
    "claude-opus-4-6-max": (7, -8, "left"),
    "gpt-5.4-medium": (-7, 6, "right"),
}

rows = [r for r in csv.DictReader(open(OUT / "summary.csv"))
        if r["is_variant"] != "True"]
assert len(rows) == 26, len(rows)

fig, ax = plt.subplots(figsize=(9.2, 8.2))
lo = min(min(float(r["B_above_lo"]) for r in rows),
         min(float(r["B_below_lo"]) for r in rows)) - 0.03
hi = max(max(float(r["B_above_hi"]) for r in rows),
         max(float(r["B_below_hi"]) for r in rows)) + 0.03
ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x (symmetric leakage)")
ax.axhline(0, color="0.75", lw=0.7, zorder=0)
ax.axvline(0, color="0.75", lw=0.7, zorder=0)

seen = set()
for r in rows:
    name = r["display"]
    x, y = float(r["B_above"]), float(r["B_below"])
    fam, c = family(name)
    ax.errorbar(x, y,
                xerr=[[x - float(r["B_above_lo"])], [float(r["B_above_hi"]) - x]],
                yerr=[[y - float(r["B_below_lo"])], [float(r["B_below_hi"]) - y]],
                fmt="o", color=c, ecolor="0.75", elinewidth=1.1, capsize=2,
                ms=6, zorder=3, label=fam if fam not in seen else None)
    seen.add(fam)
    dx, dy, ha = OFFSETS.get(name, (6, 3, "left"))
    ax.annotate(name, (x, y), textcoords="offset points", xytext=(dx, dy),
                fontsize=8, ha=ha, zorder=4)

ax.set_xlabel("bias$_{\\mathrm{above}}$  =  P(> $\\tau$ | above_good) $-$ q")
ax.set_ylabel("bias$_{\\mathrm{below}}$  =  P($\\leq \\tau$ | below_good) $-$ (1$-$q)")
ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)
ax.set_aspect("equal")
ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
ax.grid(alpha=0.2, lw=0.5)
fig.tight_layout()
fig.savefig(OUT / "fig_scatter_clean.png", dpi=200)
print("wrote", OUT / "fig_scatter_clean.png")
