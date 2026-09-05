# fine-graining-value-leakage

Analysis code for my MATS 12.0 application (Neel Nanda stream): decomposing the
Donation Bet bias metric from "Value Leakage" (Betley et al. 2026,
arXiv 2607.14345) into a good-side pull ("lean") and a direction-neutral shift
("push"), and measuring push independently in the paper's charity-pair
framings, where a donation happens on both sides of the threshold.

## Data

All analyses run against the paper's released code and rollout caches, expected
as clones under `ext/` (not committed here):

- `ext/value_leakage` — https://github.com/TruthfulAI-research/value_leakage
- `ext/value_leakage_data` — https://github.com/TruthfulAI-research/value_leakage_data

## Scripts

- `phase0_unpooled_bias.py` — main-experiment decomposition (26 models x 9
  questions x ~100 rollouts/condition): per-direction biases, lean/push,
  bootstrap CIs. Writes `phase0_out/summary.csv`, `lean_push_tables.md`,
  the bias matrices, scatter and heatmaps.
- `phase_free_f5_push.py` — framing-sweep analysis (4 models x 12 conditions):
  charity-pair push crosscheck, exact-threshold tie / IQR anchoring analysis,
  per-variant lean-push ladders. Writes the `phase0_out/f5_*` outputs.
- `phase0_sysprompt_compare.py` — system-prompt conditions (5 Claude models).
- `phase0_step3_born_made.py` — first vs. final in-CoT estimate decomposition.
- `bias_tables.py`, `illustrate_gpt55.py`, `plot_decomposition_bars.py`,
  `plot_push_heatmap.py`, `plot_scatter_clean.py` — tables and figures.
- `fresh_batch_run.py` / `fresh_batch_analyze.py` — small fresh OpenRouter
  smoke batch (not verified, not used in the write-up).

## Outputs

`phase0_out/` holds every CSV, markdown table and figure referenced in the
write-up, plus the seed-7 randomly selected transcripts
(`random_transcripts*/`).

`OPENROUTER_API_KEY` (in `.env`) is needed only by `fresh_batch_run.py`;
everything else runs offline from the caches.

*(README written with Claude assistance.)*
