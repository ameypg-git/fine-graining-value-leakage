"""Independent push measurement from the paper's released framing-sweep caches.

The paper's "Sensitivity to different framings" appendix ran, for FOUR models
(claude-opus-4.7-high, claude-opus-4.8-high, gemini-3.1-pro-high, gpt-5.5-high),
ten note variants of the Donation Bet (V1-V5 wording variants, C1-C5 charity
pairs; see donation_bet/plot_framing_bias.py) at 10 rollouts/cell across all 9
questions -- and reported ONLY the pooled bias per variant, the statistic that
cancels the direction-neutral push by construction. This script computes the
unpooled statistics from those same caches.

Definitions (per model x question x experiment; upward-shift formulation):
    tau  = threshold used in the directional prompts (from the shared baseline)
    q    = P(baseline estimate > tau)
    u_A  = P(estimate > tau | favored-side-above) - q     (upward shift, A)
    u_B  = P(estimate > tau | favored-side-below) - q     (upward shift, B)
    push = (u_A + u_B) / 2      # note's direction-neutral shift; q-dependent
    lean = (u_A - u_B) / 2      # pull toward the favored side; q cancels
    bias_above = u_A ; bias_below = -u_B ; pooled(paper) = 2*lean

Key design facts exploited:
  * All ten f5 experiments AND main_experiment_small share the SAME baseline
    cache file per (model, task) (n=10; `_VARIANT_SHARE_BASELINE`), hence the
    same tau and the same empirical q -> the contrast
    push(main_small) - push(charity pairs) is free of baseline sampling noise.
  * The C1-C5 charity pairs put a donation on BOTH sides, so the directional
    value gradient is ~0 by construction: their push is measured where value
    leakage toward "the good side" cannot operate.
  * main_experiment_accurate (n=100, "accurate" wording frame) gives the
    high-precision main-condition numbers (as in Phase 0) for the same models.

Headline tests:
  (i)  push cross-check: push(main) vs mean push over C1-C5, per model
  (ii) ladder: lean should track the value differential across variants,
       push should not (if push is note-mechanics)
  (iii) corrected per-direction value effect:
       value_above = bias_above(main) - push*, value_below = bias_below(main)
       + push*, push* = C1-C5 mean. Prediction: both >= 0, roughly equal.

Outputs under phase0_out/:
    f5_per_question.csv, f5_summary.csv, f5_crosscheck.csv,
    f5_lean_push_tables.md, fig_note_ladder.png, fig_push_crosscheck.png,
    random_transcripts_f5/ (12, seeded), f5_judge_check.csv (20, seeded)

Run cache-only; requires the extended sparse checkout (plain v1_* baselines,
v1_*_f5_* directionals, main_experiment_small* judge caches).
"""
import os
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

MODELS4 = ["claude-opus-4.7-high", "claude-opus-4.8-high",
           "gemini-3.1-pro-high", "gpt-5.5-high"]

# Paper's framing ids (donation_bet/plot_framing_bias.py) -> experiment names.
# C1-C5 are the charity pairs (donation on both sides).
EXPERIMENTS = {
    "MAIN_acc":   "main_experiment_accurate",        # n=100, accurate frame
    "MAIN_small": "main_experiment_small",           # n=10, plain frame
    "V1_good":    "main_experiment_small_f5_good",
    "V2_unicef":  "main_experiment_small_f5_unicef",
    "V3_beer":    "main_experiment_small_f5_beer",
    "V4_naked":   "main_experiment_small_f5_naked",
    "V5_pick":    "main_experiment_small_f5_pick",
    "C1_dwb_greenpeace":   "main_experiment_small_f5_dwb_greenpeace",
    "C2_redcross_unicef":  "main_experiment_small_f5_redcross_unicef",
    "C3_mnm_amf":          "main_experiment_small_f5_mnm_amf",
    "C4_feeding_good360":  "main_experiment_small_f5_feeding_good360",
    "C5_wwf_tnc":          "main_experiment_small_f5_wwf_tnc",
}
PAIR_LABELS = ["C1_dwb_greenpeace", "C2_redcross_unicef", "C3_mnm_amf",
               "C4_feeding_good360", "C5_wwf_tnc"]
SMALL_LABELS = [l for l in EXPERIMENTS if l != "MAIN_acc"]  # share baselines
LADDER_ORDER = PAIR_LABELS + ["V5_pick", "V3_beer", "V4_naked",
                              "V1_good", "V2_unicef", "MAIN_small", "MAIN_acc"]

N_BOOT = 5000
BOOT_SEED = 0
TRANSCRIPT_SEED = 7


def norm_task(pk):
    """v1_giraffes / v1_giraffes_accurate / v1_giraffes_f5_wwf_tnc -> giraffes."""
    t = pk.removeprefix("v1_")
    for task in ["bridge", "crochet", "giraffes", "maiden", "orangecars",
                 "tbc", "turns", "windowdays", "zills"]:
        if t == task or t.startswith(task + "_"):
            return task
    return t


def load_experiment(label, exp_name, model, est_store=None):
    """Return (per-question stats list, raw df) for one model x experiment.

    If est_store is given, stashes the parsed estimate arrays under
    est_store[(model, label, task)] = {"base": ..., "A": ..., "B": ...} for
    the continuous-scale (median displacement) analysis.
    """
    df, thresholds, display = get_main_dfs(
        exp_name, [model], cache_only=True, raw=True)[model]
    rows = []
    for pk in sorted(thresholds):
        t = thresholds[pk][0]
        sub = df[df.prompt_key == pk]
        stats = {}
        for direction, name in [("baseline", "base"), ("above_good", "A"),
                                ("below_good", "B")]:
            d = sub[sub.direction == direction]
            est = d.estimate.dropna().astype(float)
            stats[name] = est.to_numpy()
            stats[f"n_{name}_raw"] = len(d)
            stats[f"n_{name}"] = len(est)
        if min(stats["n_base"], stats["n_A"], stats["n_B"]) == 0:
            continue
        # thresholds recorded in the directional cache metadata must agree
        # with the recomputed one (same shared baseline) -- verify via the
        # threshold column on directional rows.
        rec = sub.loc[sub.direction != "baseline", "threshold"].dropna().unique()
        thr_mismatch = len(rec) > 0 and not np.allclose(rec.astype(float), t)
        q = float((stats["base"] > t).mean())
        pA = float((stats["A"] > t).mean())
        pB = float((stats["B"] > t).mean())
        if est_store is not None:
            est_store[(model, label, norm_task(pk))] = dict(
                base=stats["base"], A=stats["A"], B=stats["B"], t=t)
        rows.append(dict(
            label=label, experiment=exp_name, model=model, display=display,
            task=norm_task(pk), prompt_key=pk, threshold=t,
            thr_mismatch=bool(thr_mismatch),
            n_base=stats["n_base"], n_A=stats["n_A"], n_B=stats["n_B"],
            drop_base=1 - stats["n_base"] / max(stats["n_base_raw"], 1),
            drop_A=1 - stats["n_A"] / max(stats["n_A_raw"], 1),
            drop_B=1 - stats["n_B"] / max(stats["n_B_raw"], 1),
            q=q, pA=pA, pB=pB,
            u_A=pA - q, u_B=pB - q,
            push=(pA + pB) / 2 - q, lean=(pA - pB) / 2,
            bias_above=pA - q, bias_below=q - pB,
            pooled=pA - pB,
            tie_base=float((stats["base"] == t).mean()),
            tie_A=float((stats["A"] == t).mean()),
            tie_B=float((stats["B"] == t).mean()),
            # anchoring signature: mass within 10% of the quoted threshold,
            # and dispersion (IQR of relative displacement) vs baseline
            near_base=float((np.abs(stats["base"] - t) <= 0.1 * t).mean()),
            near_A=float((np.abs(stats["A"] - t) <= 0.1 * t).mean()),
            near_B=float((np.abs(stats["B"] - t) <= 0.1 * t).mean()),
            iqr_base=float(np.subtract(*np.percentile(
                (stats["base"] - t) / t, [75, 25]))),
            iqr_A=float(np.subtract(*np.percentile(
                (stats["A"] - t) / t, [75, 25]))),
            iqr_B=float(np.subtract(*np.percentile(
                (stats["B"] - t) / t, [75, 25]))),
        ))
    return rows, df


def joint_bootstrap(model_rows, n_boot=N_BOOT, seed=BOOT_SEED):
    """Joint bootstrap for one model over all experiments x tasks.

    Baseline draws are SHARED across all small-frame experiments of a task
    (they literally share the same 10 cached baseline rollouts), so contrasts
    among small-frame pushes are baseline-noise-free. MAIN_acc has its own
    independent baseline draw. Returns dict of stat arrays (n_boot,).
    """
    rng = np.random.default_rng(seed)
    by_label = {l: {} for l in EXPERIMENTS}
    for r in model_rows:
        by_label[r["label"]][r["task"]] = r
    tasks_small = sorted(set.intersection(
        *[set(by_label[l]) for l in SMALL_LABELS])) or []
    tasks_acc = sorted(by_label["MAIN_acc"])

    # shared small-frame baseline draws, keyed by task
    q_small = {}
    for task in tasks_small:
        r = by_label["MAIN_small"][task]
        q_small[task] = rng.binomial(r["n_base"], r["q"], n_boot) / r["n_base"]

    out = {}
    for label in EXPERIMENTS:
        tasks = tasks_acc if label == "MAIN_acc" else tasks_small
        push = np.zeros(n_boot)
        lean = np.zeros(n_boot)
        b_above = np.zeros(n_boot)
        b_below = np.zeros(n_boot)
        for task in tasks:
            r = by_label[label][task]
            if label == "MAIN_acc":
                qs = rng.binomial(r["n_base"], r["q"], n_boot) / r["n_base"]
            else:
                qs = q_small[task]
            pa = rng.binomial(r["n_A"], r["pA"], n_boot) / r["n_A"]
            pb = rng.binomial(r["n_B"], r["pB"], n_boot) / r["n_B"]
            push += ((pa + pb) / 2 - qs) / len(tasks)
            lean += ((pa - pb) / 2) / len(tasks)
            b_above += (pa - qs) / len(tasks)
            b_below += (qs - pb) / len(tasks)
        out[label] = dict(push=push, lean=lean,
                          bias_above=b_above, bias_below=b_below)
    out["push_pairs"] = np.mean([out[l]["push"] for l in PAIR_LABELS], axis=0)
    out["lean_pairs_abs"] = np.mean(
        [np.abs(out[l]["lean"]) for l in PAIR_LABELS], axis=0)
    out["contrast_small"] = out["MAIN_small"]["push"] - out["push_pairs"]
    out["contrast_acc"] = out["MAIN_acc"]["push"] - out["push_pairs"]
    for frame in ["MAIN_small", "MAIN_acc"]:
        out[f"value_above_{frame}"] = out[frame]["bias_above"] - out["push_pairs"]
        out[f"value_below_{frame}"] = out[frame]["bias_below"] + out["push_pairs"]
        out[f"value_gap_{frame}"] = (out[f"value_above_{frame}"]
                                     - out[f"value_below_{frame}"])
    return out


def joint_bootstrap_cont(est_store, model, n_boot=N_BOOT, seed=BOOT_SEED + 1):
    """Continuous-scale twin of joint_bootstrap.

    Outcome per cell = median of (estimate - tau)/tau (the paper's App. E.5
    displacement unit). Unlike P(> tau), this scale has NO ceiling: a large
    lean cannot mechanically squeeze the measurable push to zero. Baseline
    resample indices are shared across the small-frame experiments of a task
    (same 10 cached baseline rollouts).
    """
    rng = np.random.default_rng(seed)
    by = {l: {} for l in EXPERIMENTS}
    for (m, l, task), est in est_store.items():
        if m == model:
            by[l][task] = est
    tasks_small = sorted(set.intersection(*[set(by[l]) for l in SMALL_LABELS]))
    tasks_acc = sorted(by["MAIN_acc"])

    def med_boot(arr):
        idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
        return np.median(arr[idx], axis=1)

    base_small = {}
    for task in tasks_small:
        est = by["MAIN_small"][task]
        base_small[task] = med_boot((est["base"] - est["t"]) / est["t"])

    out = {}
    for label in EXPERIMENTS:
        tasks = tasks_acc if label == "MAIN_acc" else tasks_small
        push = np.zeros(n_boot)
        lean = np.zeros(n_boot)
        for task in tasks:
            est = by[label][task]
            t = est["t"]
            mb = (med_boot((est["base"] - t) / t) if label == "MAIN_acc"
                  else base_small[task])
            mA = med_boot((est["A"] - t) / t)
            mB = med_boot((est["B"] - t) / t)
            push += ((mA + mB) / 2 - mb) / len(tasks)
            lean += ((mA - mB) / 2) / len(tasks)
        out[label] = dict(push=push, lean=lean)
    out["push_pairs"] = np.mean([out[l]["push"] for l in PAIR_LABELS], axis=0)
    out["contrast_small"] = out["MAIN_small"]["push"] - out["push_pairs"]
    out["contrast_acc"] = out["MAIN_acc"]["push"] - out["push_pairs"]
    return out


def cont_point(est_store, model):
    """Point estimates on the continuous scale: {(label): {push, lean}}."""
    by = {l: {} for l in EXPERIMENTS}
    for (m, l, task), est in est_store.items():
        if m == model:
            by[l][task] = est
    pt = {}
    for label, tasks in by.items():
        if not tasks:
            continue
        pushes, leans = [], []
        for est in tasks.values():
            t = est["t"]
            mb = float(np.median((est["base"] - t) / t))
            mA = float(np.median((est["A"] - t) / t))
            mB = float(np.median((est["B"] - t) / t))
            pushes.append((mA + mB) / 2 - mb)
            leans.append((mA - mB) / 2)
        pt[label] = dict(push=float(np.mean(pushes)),
                         lean=float(np.mean(leans)), n_tasks=len(tasks))
    return pt


def ci(a):
    lo, hi = np.quantile(a, (0.025, 0.975))
    return float(lo), float(hi)


def fmt(x, lo, hi, star_if_excl0=True):
    s = "*" if star_if_excl0 and (lo > 0 or hi < 0) else " "
    return f"{x:+.3f} [{lo:+.3f},{hi:+.3f}]{s}"


def main():
    OUT.mkdir(exist_ok=True)
    per_q, failures = [], []
    frames = {}          # (model, label) -> raw df, for transcripts/judge check
    point = {}           # (model, label) -> equal-task means
    boots = {}           # model -> joint bootstrap dict
    est_store = {}       # (model, label, task) -> estimate arrays + tau
    point_c = {}         # model -> {label: continuous push/lean}
    boots_c = {}         # model -> continuous bootstrap dict

    for model in MODELS4:
        model_rows = []
        for label, exp_name in EXPERIMENTS.items():
            try:
                rows, df = load_experiment(label, exp_name, model, est_store)
            except Exception as e:
                failures.append((model, label, repr(e)[:160]))
                continue
            frames[(model, label)] = df
            model_rows.extend(rows)
            per_q.extend(rows)
            if rows:
                rdf = pd.DataFrame(rows)
                point[(model, label)] = dict(
                    n_tasks=len(rdf),
                    push=rdf.push.mean(), lean=rdf.lean.mean(),
                    bias_above=rdf.bias_above.mean(),
                    bias_below=rdf.bias_below.mean(),
                    pooled=rdf.pooled.mean(),
                    drop_A=rdf.drop_A.mean(), drop_B=rdf.drop_B.mean(),
                    drop_base=rdf.drop_base.mean(),
                    n_A=rdf.n_A.sum(), n_B=rdf.n_B.sum(),
                    # fraction of questions where the outcome scale is pinned
                    # (favored side taken >=95% of the time in either
                    # direction): there, push has no room to show up in
                    # P(> tau) and the continuous scale is the one to trust.
                    frac_saturated=float(
                        ((rdf.pA >= 0.95) | (rdf.pB <= 0.05)).mean()),
                )
        boots[model] = joint_bootstrap(model_rows)
        boots_c[model] = joint_bootstrap_cont(est_store, model)
        point_c[model] = cont_point(est_store, model)

    pq_df = pd.DataFrame(per_q)
    pq_df.to_csv(OUT / "f5_per_question.csv", index=False)

    if failures:
        print("FAILURES (cache misses etc.):")
        for m, l, e in failures:
            print(f"  {m} {l}: {e}")

    # ---- consistency checks -------------------------------------------------
    print("\n=== Consistency checks ===")
    bad_thr = pq_df[pq_df.thr_mismatch]
    print(f"threshold recomputed==recorded: "
          f"{'OK' if bad_thr.empty else f'{len(bad_thr)} MISMATCHES'}")
    # same tau and q across all small-frame experiments of a (model, task)
    small = pq_df[pq_df.label != "MAIN_acc"]
    tq = small.groupby(["model", "task"])[["threshold", "q"]].nunique()
    n_bad = int((tq > 1).any(axis=1).sum())
    print(f"shared baseline => identical (tau, q) across small experiments: "
          f"{'OK' if n_bad == 0 else f'{n_bad} (model,task) VIOLATIONS'}")
    acc = pq_df[pq_df.label == "MAIN_acc"][["model", "task", "threshold"]]
    mrg = acc.merge(
        small[small.label == "MAIN_small"][["model", "task", "threshold"]],
        on=["model", "task"], suffixes=("_acc", "_small"))
    diff_frac = (mrg.threshold_acc != mrg.threshold_small).mean()
    print(f"tau differs between accurate and plain frames on "
          f"{diff_frac:.0%} of (model,task) cells (expected; frames have "
          f"their own baselines)")

    # ---- summary table ------------------------------------------------------
    sum_rows = []
    for model in MODELS4:
        for label in EXPERIMENTS:
            if (model, label) not in point:
                continue
            p = point[(model, label)]
            b = boots[model][label]
            sum_rows.append(dict(
                model=model, label=label, **{k: p[k] for k in p},
                push_lo=ci(b["push"])[0], push_hi=ci(b["push"])[1],
                lean_lo=ci(b["lean"])[0], lean_hi=ci(b["lean"])[1],
            ))
    sum_df = pd.DataFrame(sum_rows)
    sum_df.to_csv(OUT / "f5_summary.csv", index=False)

    # ---- cross-check table --------------------------------------------------
    xc_rows = []
    for model in MODELS4:
        b = boots[model]
        pairs_pt = np.mean([point[(model, l)]["push"] for l in PAIR_LABELS])
        row = dict(
            model=model,
            push_pairs=pairs_pt,
            push_pairs_lo=ci(b["push_pairs"])[0],
            push_pairs_hi=ci(b["push_pairs"])[1],
            push_main_small=point[(model, "MAIN_small")]["push"],
            push_main_small_lo=ci(b["MAIN_small"]["push"])[0],
            push_main_small_hi=ci(b["MAIN_small"]["push"])[1],
            push_main_acc=point[(model, "MAIN_acc")]["push"],
            push_main_acc_lo=ci(b["MAIN_acc"]["push"])[0],
            push_main_acc_hi=ci(b["MAIN_acc"]["push"])[1],
            contrast_small=float(np.mean(b["contrast_small"])),
            contrast_small_lo=ci(b["contrast_small"])[0],
            contrast_small_hi=ci(b["contrast_small"])[1],
            contrast_acc=float(np.mean(b["contrast_acc"])),
            contrast_acc_lo=ci(b["contrast_acc"])[0],
            contrast_acc_hi=ci(b["contrast_acc"])[1],
        )
        for frame in ["MAIN_small", "MAIN_acc"]:
            for stat in ["value_above", "value_below", "value_gap"]:
                arr = b[f"{stat}_{frame}"]
                row[f"{stat}_{frame}"] = float(np.mean(arr))
                row[f"{stat}_{frame}_lo"], row[f"{stat}_{frame}_hi"] = ci(arr)
        xc_rows.append(row)
    xc_df = pd.DataFrame(xc_rows)
    xc_df.to_csv(OUT / "f5_crosscheck.csv", index=False)

    # ---- printed report -----------------------------------------------------
    print("\n=== Test (i): push cross-check (per model, equal-task mean, 95% CI) ===")
    print("push_pairs = mean over C1-C5 (donation on BOTH sides: no good side)")
    for _, r in xc_df.iterrows():
        print(f"\n  {r.model}")
        print(f"    push  C1-C5 pairs      : "
              f"{fmt(r.push_pairs, r.push_pairs_lo, r.push_pairs_hi)}")
        print(f"    push  MAIN small frame : "
              f"{fmt(r.push_main_small, r.push_main_small_lo, r.push_main_small_hi)}")
        print(f"    push  MAIN accurate    : "
              f"{fmt(r.push_main_acc, r.push_main_acc_lo, r.push_main_acc_hi)}")
        print(f"    contrast small - pairs : "
              f"{fmt(r.contrast_small, r.contrast_small_lo, r.contrast_small_hi)}"
              f"   <- baseline-noise-free")
        print(f"    contrast acc   - pairs : "
              f"{fmt(r.contrast_acc, r.contrast_acc_lo, r.contrast_acc_hi)}")

    print("\n=== Test (i) on the CONTINUOUS scale "
          "(median (est-tau)/tau displacement vs baseline; no ceiling) ===")
    xc_c_rows = []
    for model in MODELS4:
        bc = boots_c[model]
        pc = point_c[model]
        pairs_pt = float(np.mean([pc[l]["push"] for l in PAIR_LABELS]))
        row = dict(
            model=model, push_pairs=pairs_pt,
            push_pairs_lo=ci(bc["push_pairs"])[0],
            push_pairs_hi=ci(bc["push_pairs"])[1],
            push_main_small=pc["MAIN_small"]["push"],
            push_main_small_lo=ci(bc["MAIN_small"]["push"])[0],
            push_main_small_hi=ci(bc["MAIN_small"]["push"])[1],
            push_main_acc=pc["MAIN_acc"]["push"],
            push_main_acc_lo=ci(bc["MAIN_acc"]["push"])[0],
            push_main_acc_hi=ci(bc["MAIN_acc"]["push"])[1],
            contrast_small=float(np.mean(bc["contrast_small"])),
            contrast_small_lo=ci(bc["contrast_small"])[0],
            contrast_small_hi=ci(bc["contrast_small"])[1],
            contrast_acc=float(np.mean(bc["contrast_acc"])),
            contrast_acc_lo=ci(bc["contrast_acc"])[0],
            contrast_acc_hi=ci(bc["contrast_acc"])[1],
        )
        xc_c_rows.append(row)
        print(f"\n  {model}")
        print(f"    push  C1-C5 pairs      : "
              f"{fmt(row['push_pairs'], row['push_pairs_lo'], row['push_pairs_hi'])}")
        print(f"    push  MAIN small frame : "
              f"{fmt(row['push_main_small'], row['push_main_small_lo'], row['push_main_small_hi'])}")
        print(f"    push  MAIN accurate    : "
              f"{fmt(row['push_main_acc'], row['push_main_acc_lo'], row['push_main_acc_hi'])}")
        print(f"    contrast small - pairs : "
              f"{fmt(row['contrast_small'], row['contrast_small_lo'], row['contrast_small_hi'])}")
        print(f"    contrast acc   - pairs : "
              f"{fmt(row['contrast_acc'], row['contrast_acc_lo'], row['contrast_acc_hi'])}")
    xc_c_df = pd.DataFrame(xc_c_rows)
    xc_c_df.to_csv(OUT / "f5_crosscheck_continuous.csv", index=False)

    print("\n=== Test (iii): corrected per-direction value effect "
          "(push* = C1-C5 mean) ===")
    for _, r in xc_df.iterrows():
        print(f"  {r.model}")
        for frame in ["MAIN_acc", "MAIN_small"]:
            va = fmt(r[f"value_above_{frame}"],
                     r[f"value_above_{frame}_lo"], r[f"value_above_{frame}_hi"])
            vb = fmt(r[f"value_below_{frame}"],
                     r[f"value_below_{frame}_lo"], r[f"value_below_{frame}_hi"])
            vg = fmt(r[f"value_gap_{frame}"],
                     r[f"value_gap_{frame}_lo"], r[f"value_gap_{frame}_hi"])
            print(f"    {frame:10s} value_above={va}  value_below={vb}  gap={vg}")

    print("\n=== Ladder (test ii): lean and push per note variant ===")
    for model in MODELS4:
        print(f"\n  {model}  (per-cell n from caches; drops shown if >2%)")
        for label in LADDER_ORDER:
            if (model, label) not in point:
                continue
            p = point[(model, label)]
            b = boots[model][label]
            drops = ""
            if max(p["drop_A"], p["drop_B"]) > 0.02:
                drops = f"  drops A/B={p['drop_A']:.0%}/{p['drop_B']:.0%}"
            sat = (f"  SAT={p['frac_saturated']:.0%}"
                   if p["frac_saturated"] >= 0.3 else "")
            pc = point_c[model].get(label)
            cont = (f"  | cont: lean={pc['lean']:+.2f} push={pc['push']:+.2f}"
                    if pc else "")
            print(f"    {label:20s} lean={fmt(p['lean'], *ci(b['lean']))}  "
                  f"push={fmt(p['push'], *ci(b['push']))}{sat}{drops}{cont}")

    print("\n=== Anchoring signature: do answers concentrate at/near the "
          "quoted threshold? ===")
    print("near10 = fraction of answers within 10% of tau; tie = exactly tau;")
    print("IQR ratio = spread of (est-tau)/tau in note conditions / baseline "
          "(<1 = compression toward the anchor). Directions pooled.")
    print(f"  {'model':22s} {'condition':12s} {'near10 base->note':>18s} "
          f"{'tie base->note':>15s} {'IQR ratio':>10s}")
    for model in MODELS4:
        groups = {
            "MAIN_acc": ["MAIN_acc"], "MAIN_small": ["MAIN_small"],
            "C pairs": PAIR_LABELS,
            "V variants": ["V1_good", "V2_unicef", "V3_beer", "V4_naked",
                           "V5_pick"],
        }
        for gname, labels in groups.items():
            sub = pq_df[(pq_df.model == model) & pq_df.label.isin(labels)]
            if sub.empty:
                continue
            near_note = (sub.near_A.mean() + sub.near_B.mean()) / 2
            tie_note = (sub.tie_A.mean() + sub.tie_B.mean()) / 2
            iqr_note = (sub.iqr_A.mean() + sub.iqr_B.mean()) / 2
            iqr_ratio = iqr_note / max(sub.iqr_base.mean(), 1e-9)
            print(f"  {model:22s} {gname:12s} "
                  f"{sub.near_base.mean():>7.0%} -> {near_note:<7.0%} "
                  f"{sub.tie_base.mean():>6.0%} -> {tie_note:<6.0%} "
                  f"{iqr_ratio:>9.2f}")

    print("\n=== Reproduction anchor: paper's pooled bias per framing "
          "(= 2*lean; compare to framing appendix bars, C2/C3/C5 flipped "
          "there) ===")
    for model in MODELS4:
        vals = []
        for label in LADDER_ORDER:
            if (model, label) in point:
                vals.append(f"{label.split('_')[0]}={point[(model,label)]['pooled']:+.2f}")
        print(f"  {model}: " + "  ".join(vals))

    # ---- figures ------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ladder figure
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for ax, model in zip(axes.flat, MODELS4):
        labels = [l for l in LADDER_ORDER if (model, l) in point]
        x = np.arange(len(labels))
        for stat, color, mk in [("lean", "#1f77b4", "o"), ("push", "#d62728", "s")]:
            y = [point[(model, l)][stat] for l in labels]
            los, his = zip(*[ci(boots[model][l][stat]) for l in labels])
            yerr = [np.array(y) - np.array(los), np.array(his) - np.array(y)]
            ax.errorbar(x, y, yerr=yerr, fmt=mk + "-", ms=4, lw=1,
                        capsize=2, label=stat, color=color)
        ax.axhline(0, color="gray", lw=0.6)
        ax.axvline(4.5, color="gray", lw=0.6, ls=":")
        ax.text(2, ax.get_ylim()[1] * 0.92, "no good side", fontsize=7,
                ha="center", color="gray")
        ax.set_title(model, fontsize=10)
        ax.set_xticks(x, [l.replace("_", "\n", 1) for l in labels],
                      rotation=45, ha="right", fontsize=7)
        ax.legend(fontsize=8)
    fig.suptitle("Note ladder: lean moves with the value differential; "
                 "does push stay put?", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig_note_ladder.png", dpi=200)

    # cross-check scatter
    fig2, ax2 = plt.subplots(figsize=(6.5, 6))
    for _, r in xc_df.iterrows():
        ax2.errorbar(
            r.push_main_small, r.push_pairs,
            xerr=[[r.push_main_small - r.push_main_small_lo],
                  [r.push_main_small_hi - r.push_main_small]],
            yerr=[[r.push_pairs - r.push_pairs_lo],
                  [r.push_pairs_hi - r.push_pairs]],
            fmt="o", ms=6, capsize=3, label=f"{r.model} (plain frame)")
        ax2.errorbar(
            r.push_main_acc, r.push_pairs,
            xerr=[[r.push_main_acc - r.push_main_acc_lo],
                  [r.push_main_acc_hi - r.push_main_acc]],
            yerr=[[r.push_pairs - r.push_pairs_lo],
                  [r.push_pairs_hi - r.push_pairs]],
            fmt="^", ms=6, capsize=3, alpha=0.6,
            label=f"{r.model} (accurate frame)")
    lims = [-0.35, 0.35]
    ax2.plot(lims, lims, "k--", lw=0.8)
    ax2.axhline(0, color="gray", lw=0.5)
    ax2.axvline(0, color="gray", lw=0.5)
    ax2.set_xlim(lims); ax2.set_ylim(lims)
    ax2.set_xlabel("push in MAIN bet (charity vs nothing)")
    ax2.set_ylabel("push in charity-pair bets (no good side)")
    ax2.set_title("Is the main-condition asymmetry a value-free push?\n"
                  "on y=x: yes, fully (outcome scale; ceiling-limited "
                  "where lean saturates)")
    ax2.legend(fontsize=7)
    fig2.tight_layout()
    fig2.savefig(OUT / "fig_push_crosscheck.png", dpi=200)

    # continuous-scale ladder figure
    fig3, axes3 = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for ax, model in zip(axes3.flat, MODELS4):
        labels = [l for l in LADDER_ORDER if l in point_c[model]]
        x = np.arange(len(labels))
        for stat, color, mk in [("lean", "#1f77b4", "o"), ("push", "#d62728", "s")]:
            y = [point_c[model][l][stat] for l in labels]
            los, his = zip(*[ci(boots_c[model][l][stat]) for l in labels])
            yerr = [np.array(y) - np.array(los), np.array(his) - np.array(y)]
            ax.errorbar(x, y, yerr=yerr, fmt=mk + "-", ms=4, lw=1,
                        capsize=2, label=stat, color=color)
        ax.axhline(0, color="gray", lw=0.6)
        ax.axvline(4.5, color="gray", lw=0.6, ls=":")
        ax.set_title(model, fontsize=10)
        ax.set_xticks(x, [l.replace("_", "\n", 1) for l in labels],
                      rotation=45, ha="right", fontsize=7)
        ax.legend(fontsize=8)
        ax.set_ylabel("median (est−τ)/τ vs baseline")
    fig3.suptitle("Note ladder on the CONTINUOUS scale (no ceiling): "
                  "median relative displacement", fontsize=12)
    fig3.tight_layout()
    fig3.savefig(OUT / "fig_note_ladder_continuous.png", dpi=200)

    # ---- human-verification artifacts --------------------------------------
    rng = np.random.default_rng(TRANSCRIPT_SEED)
    tdir = OUT / "random_transcripts_f5"
    tdir.mkdir(exist_ok=True)
    # ensure coverage: naked, beer, one pair, main_small + random rest
    forced = ["V4_naked", "V3_beer", "C5_wwf_tnc", "MAIN_small"]
    picks = []
    for i in range(12):
        model = MODELS4[int(rng.integers(len(MODELS4)))]
        label = forced[i % len(forced)] if i < 8 else \
            list(EXPERIMENTS)[int(rng.integers(len(EXPERIMENTS)))]
        df = frames.get((model, label))
        if df is None or df.empty:
            continue
        row = df.iloc[int(rng.integers(len(df)))]
        picks.append((model, label, row))
    for i, (model, label, row) in enumerate(picks):
        est = row.estimate
        with open(tdir / f"{i:02d}_{model}_{label}_{row.prompt_key}_{row.direction}.txt",
                  "w") as f:
            f.write(f"model={model} experiment={label} prompt={row.prompt_key} "
                    f"direction={row.direction}\n"
                    f"threshold={row.threshold} judge_estimate={est}\n"
                    f"{'='*70}\nPROMPT:\n{row.prompt}\n"
                    f"{'='*70}\nREASONING:\n{row.reasoning}\n"
                    f"{'='*70}\nANSWER:\n{row.answer}\n")

    jc = []
    for i in range(20):
        model = MODELS4[int(rng.integers(len(MODELS4)))]
        label = list(EXPERIMENTS)[int(rng.integers(len(EXPERIMENTS)))]
        df = frames.get((model, label))
        if df is None or df.empty:
            continue
        row = df.iloc[int(rng.integers(len(df)))]
        jc.append(dict(model=model, label=label, prompt_key=row.prompt_key,
                       direction=row.direction, judge_estimate=row.estimate,
                       threshold=row.threshold,
                       answer_text=str(row.answer)[:2000]))
    pd.DataFrame(jc).to_csv(OUT / "f5_judge_check.csv", index=False)

    # ---- markdown tables ----------------------------------------------------
    md = ["# Independent push measurement — framing-sweep caches (n=10/cell)",
          "",
          "push = (u_A + u_B)/2 (direction-neutral shift), "
          "lean = (u_A − u_B)/2 (pull toward favored side); "
          "u = P(estimate > τ) − q vs the shared baseline.",
          "C1–C5 put a donation on both sides → no good side → their push is "
          "measured where value leakage cannot operate.",
          "MAIN_small shares the C-variants' baseline rollouts: the "
          "(MAIN_small − pairs) contrast is baseline-noise-free.",
          "* = 95% bootstrap CI excludes 0 (clustered by question, joint "
          "draws, seed 0).", ""]
    md += ["## Test (i): push cross-check", "",
           "| model | push C1–C5 | push MAIN small | push MAIN accurate | "
           "Δ(small−pairs) | Δ(acc−pairs) |", "|---|---|---|---|---|---|"]
    for _, r in xc_df.iterrows():
        md.append(
            f"| {r.model} | {fmt(r.push_pairs, r.push_pairs_lo, r.push_pairs_hi)} "
            f"| {fmt(r.push_main_small, r.push_main_small_lo, r.push_main_small_hi)} "
            f"| {fmt(r.push_main_acc, r.push_main_acc_lo, r.push_main_acc_hi)} "
            f"| {fmt(r.contrast_small, r.contrast_small_lo, r.contrast_small_hi)} "
            f"| {fmt(r.contrast_acc, r.contrast_acc_lo, r.contrast_acc_hi)} |")
    md += ["", "## Test (iii): corrected per-direction value effects "
           "(push* = C1–C5 mean)", "",
           "| model | frame | value_above | value_below | gap (above−below) |",
           "|---|---|---|---|---|"]
    for _, r in xc_df.iterrows():
        for frame in ["MAIN_acc", "MAIN_small"]:
            md.append(
                f"| {r.model} | {frame} "
                f"| {fmt(r[f'value_above_{frame}'], r[f'value_above_{frame}_lo'], r[f'value_above_{frame}_hi'])} "
                f"| {fmt(r[f'value_below_{frame}'], r[f'value_below_{frame}_lo'], r[f'value_below_{frame}_hi'])} "
                f"| {fmt(r[f'value_gap_{frame}'], r[f'value_gap_{frame}_lo'], r[f'value_gap_{frame}_hi'])} |")
    md += ["", "## Test (i) on the continuous scale "
           "(median (est−τ)/τ displacement; no ceiling)", "",
           "The outcome scale P(>τ) is bounded: when lean saturates "
           "(favored side ≥95% — SAT column below), push has no room to "
           "register there and the continuous scale is the one to trust.", "",
           "| model | push C1–C5 | push MAIN small | push MAIN accurate | "
           "Δ(small−pairs) | Δ(acc−pairs) |", "|---|---|---|---|---|---|"]
    for _, r in xc_c_df.iterrows():
        md.append(
            f"| {r.model} | {fmt(r.push_pairs, r.push_pairs_lo, r.push_pairs_hi)} "
            f"| {fmt(r.push_main_small, r.push_main_small_lo, r.push_main_small_hi)} "
            f"| {fmt(r.push_main_acc, r.push_main_acc_lo, r.push_main_acc_hi)} "
            f"| {fmt(r.contrast_small, r.contrast_small_lo, r.contrast_small_hi)} "
            f"| {fmt(r.contrast_acc, r.contrast_acc_lo, r.contrast_acc_hi)} |")
    md += ["", "## Ladder: lean & push per variant", ""]
    for model in MODELS4:
        md += [f"### {model}", "",
               "| variant | lean | push | SAT | cont. lean | cont. push "
               "| pooled (=2·lean) | drops A/B |",
               "|---|---|---|---|---|---|---|---|"]
        for label in LADDER_ORDER:
            if (model, label) not in point:
                continue
            p = point[(model, label)]
            b = boots[model][label]
            pc = point_c[model].get(label, {})
            bc = boots_c[model].get(label, {})
            cl = (fmt(pc["lean"], *ci(bc["lean"])) if pc else "")
            cp = (fmt(pc["push"], *ci(bc["push"])) if pc else "")
            md.append(
                f"| {label} | {fmt(p['lean'], *ci(b['lean']))} "
                f"| {fmt(p['push'], *ci(b['push']))} "
                f"| {p['frac_saturated']:.0%} "
                f"| {cl} | {cp} "
                f"| {p['pooled']:+.3f} "
                f"| {p['drop_A']:.0%}/{p['drop_B']:.0%} |")
        md.append("")
    (OUT / "f5_lean_push_tables.md").write_text("\n".join(md))

    print(f"\nWrote: {OUT}/f5_per_question.csv, f5_summary.csv, "
          f"f5_crosscheck.csv, f5_lean_push_tables.md, fig_note_ladder.png, "
          f"fig_push_crosscheck.png, random_transcripts_f5/ "
          f"({len(picks)}, seed {TRANSCRIPT_SEED}), f5_judge_check.csv")


if __name__ == "__main__":
    main()
