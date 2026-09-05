# Per-question push heatmap: same data as fig_delta_heatmap.png but in the
# doc's own coordinate, push_q = (bias_above - bias_below)/2 = -delta/2, so the
# figure matches eq (1.1) / Table 1 without a sign flip. Rows sorted by mean
# push (descending: strongest upward push on top == same row order as the
# delta heatmap, which sorts ascending in delta).
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).parent / "phase0_out"


with open(OUT / "summary.csv") as f:
    main_models = {r["display"] for r in csv.DictReader(f)
                   if r["is_variant"] != "True"}


def load(fn):
    with open(OUT / fn) as f:
        rows = [r for r in csv.DictReader(f) if r["display"] in main_models]
    qs = [k for k in rows[0] if k not in ("display", "mean")]
    names = [r["display"] for r in rows]
    m = np.array([[float(r[q]) for q in qs] for r in rows])
    return names, qs, m


names_a, qs, above = load("bias_above_matrix.csv")
names_b, qs_b, below = load("bias_below_matrix.csv")
assert names_a == names_b and qs == qs_b
assert len(names_a) == 26, f"expected 26 main models, got {len(names_a)}"

push = (above - below) / 2.0
order = np.argsort(-push.mean(axis=1))
push, names = push[order], [names_a[i] for i in order]

fig, ax = plt.subplots(figsize=(7.2, 10.5))
im = ax.imshow(push, cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")
ax.set_xticks(range(len(qs)))
ax.set_xticklabels(qs, rotation=45, ha="right", fontsize=9)
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=9)
ax.set_title("Per-question push (red: shifted above the threshold)", fontsize=12)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, shrink=0.9)
cb.set_label("push = (bias$_{\\mathrm{above}}$ $-$ bias$_{\\mathrm{below}}$)/2", fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "fig_push_heatmap.png", dpi=200)
print("wrote", OUT / "fig_push_heatmap.png")
