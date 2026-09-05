"""Phase 1: fresh Donation Bet conversations via OpenRouter.

Reuses the paper's own pipeline (shared.runner.run_thresholds_experiment:
baseline rollouts -> judge extraction -> threshold = median of FRESH baselines
-> above_good/below_good rollouts -> judge extraction), with two patches:

  1. a new "openrouter" backend for shared.runner._create_sender
     (chat-completions against openrouter.ai, usage/cost tracking);
  2. the estimate judge routed through OpenRouter using the same judge model
     family as the paper (claude-sonnet-4.6, temperature 0).

Caches land under fresh_batch/final_data/ (never touching the paper's data).
Usage + measured cost are written to fresh_batch/logs/.

Usage:
    python3 fresh_batch_run.py --model or-qwen3.5-35 --experiment smoke_accurate
    python3 fresh_batch_run.py --model or-qwen3.5-35 --experiment main_experiment_accurate

The OpenRouter key is read from OPENROUTER_API_KEY (env), or .env files at
value-leakage/.env or the repo root. Never printed.
"""
import argparse
import json
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")

HERE = Path(__file__).resolve().parent
REPO = HERE / "ext" / "value_leakage"
FRESH = HERE / "fresh_batch" / "final_data"
LOGS = HERE / "fresh_batch" / "logs"
sys.path.insert(0, str(REPO))

# --- .env loading (OPENROUTER_API_KEY) ---
for envp in (HERE / ".env", REPO / ".env"):
    if envp.exists():
        for line in envp.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

import shared.runner as runner  # noqa: E402
from shared.models import MODELS  # noqa: E402
from shared.experiments import THRESHOLD_EXPERIMENTS  # noqa: E402
from shared.get_main_dfs import get_main_dfs  # noqa: E402

runner.CACHE_DIR = str(FRESH / "cache")
runner.ESTIMATE_JUDGE_CACHE_ROOT = str(FRESH / "estimate_judge_cache")

# --- usage / cost tracking ---
_usage_lock = threading.Lock()
USAGE = {}


def _track(mid, usage_obj):
    row = USAGE.setdefault(mid, {"requests": 0, "in_tokens": 0,
                                 "out_tokens": 0, "cost_usd": 0.0})
    d = {}
    if usage_obj is not None:
        try:
            d = usage_obj.model_dump()
        except Exception:
            d = getattr(usage_obj, "__dict__", {}) or {}
    with _usage_lock:
        row["requests"] += 1
        row["in_tokens"] += d.get("prompt_tokens") or 0
        row["out_tokens"] += d.get("completion_tokens") or 0
        row["cost_usd"] += d.get("cost") or 0.0


# --- OpenRouter backend ---
def _openrouter_sender(model):
    import openai
    from openai import OpenAI

    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY not set. Put it in value-leakage/.env as\n"
            "OPENROUTER_API_KEY=sk-or-...  (do not commit or share it)")
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key,
                    timeout=float(os.environ.get("OPENROUTER_TIMEOUT", "600")),
                    max_retries=0)

    def send(prompt):
        kwargs = dict(
            model=model["model"],
            messages=[{"role": "user", "content": prompt}],
            max_tokens=model["max_tokens"],
        )
        if not model.get("no_temperature"):
            kwargs["temperature"] = model["temperature"]
        extra = {"usage": {"include": True}}
        if model.get("reasoning_effort"):
            extra["reasoning"] = {"effort": model["reasoning_effort"]}
        last = None
        for attempt in range(7):
            try:
                resp = client.chat.completions.create(extra_body=extra, **kwargs)
                if not getattr(resp, "choices", None):
                    raise openai.APIConnectionError(request=None)  # provider hiccup -> retry
                msg = resp.choices[0].message
                md = {}
                try:
                    md = msg.model_dump()
                except Exception:
                    pass
                reasoning = (md.get("reasoning") or md.get("reasoning_content")
                             or "")
                if not reasoning and md.get("reasoning_details"):
                    parts = [p.get("text") or p.get("summary") or ""
                             for p in md["reasoning_details"]
                             if isinstance(p, dict)]
                    reasoning = "\n".join(p for p in parts if p)
                _track(model["model"], getattr(resp, "usage", None))
                return {"reasoning": reasoning or "",
                        "answer": msg.content or "", "prompt": prompt}
            except openai.APITimeoutError:
                return {"reasoning": "", "answer": "[DROPPED_API_TIMEOUT]",
                        "prompt": prompt, "blocked": True}
            except openai.BadRequestError as e:
                # some providers reject temperature on reasoning models:
                # strip it once and retry
                if "temperature" in str(e) and "temperature" in kwargs:
                    kwargs.pop("temperature")
                    last = e
                    continue
                raise
            except (openai.RateLimitError, openai.APIConnectionError,
                    openai.InternalServerError) as e:
                last = e
                time.sleep(min(2 ** attempt * 2, 60))
            except openai.APIStatusError as e:
                if getattr(e, "status_code", None) == 402:
                    body = str(e)
                    if "in_flight_budget" in body:
                        # concurrency-vs-balance limit: wait and retry
                        last = e
                        time.sleep(min(30 * (attempt + 1), 150))
                        continue
                    raise RuntimeError(
                        "OpenRouter says the account is OUT OF CREDITS. "
                        "Add credits at https://openrouter.ai/settings/credits "
                        "and re-run.") from e
                raise
        raise last

    return send


_orig_create_sender = runner._create_sender


def _create_sender(model):
    if model.get("backend") == "openrouter":
        return _openrouter_sender(model)
    return _orig_create_sender(model)


runner._create_sender = _create_sender

# --- fresh model registry (cache dir name = registry key) ---
FRESH_MODELS = {
    "or-qwen3.5-35": {
        "backend": "openrouter", "model": "qwen/qwen3.5-35b-a3b",
        "display_name": "Qwen3.5-35B-A3B (fresh)", "max_tokens": 16000,
        "temperature": 1, "max_concurrent": 30,
    },
    "or-qwen3.6-35": {
        "backend": "openrouter", "model": "qwen/qwen3.6-35b-a3b",
        "display_name": "Qwen3.6-35B-A3B (fresh)", "max_tokens": 16000,
        "temperature": 1, "max_concurrent": 30,
    },
    "or-gpt-5.5-medium": {
        "backend": "openrouter", "model": "openai/gpt-5.5",
        "display_name": "gpt-5.5-medium (fresh)", "max_tokens": 16000,
        "temperature": 1, "max_concurrent": 30,
        "reasoning_effort": "medium",
    },
}
MODELS.update(FRESH_MODELS)

# same judge model family as the paper (claude-sonnet-4-6, temp 0),
# transported via OpenRouter
runner.ESTIMATE_JUDGE_CONFIG = {
    "backend": "openrouter", "model": "anthropic/claude-sonnet-4.6",
    "max_tokens": 1024, "temperature": 0, "max_concurrent": 40,
}

# smoke experiment: 1 question, 20 per cell
THRESHOLD_EXPERIMENTS["smoke_accurate"] = {
    "prompts": ["v1_giraffes_accurate"],
    "n_baseline": 20, "n_per_threshold": 20,
    "thresholds": {"pct_range": (50, 50), "n": 1},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(FRESH_MODELS))
    ap.add_argument("--experiment", default="smoke_accurate",
                    choices=["smoke_accurate", "main_experiment_accurate"])
    ap.add_argument("--cache-only", action="store_true")
    ap.add_argument("--concurrency", type=int, default=None,
                    help="Override max_concurrent (lower this on a "
                         "low-credit account: in-flight budget scales "
                         "with balance).")
    args = ap.parse_args()
    if args.concurrency:
        MODELS[args.model]["max_concurrent"] = args.concurrency
        runner.ESTIMATE_JUDGE_CONFIG["max_concurrent"] = min(
            args.concurrency, runner.ESTIMATE_JUDGE_CONFIG["max_concurrent"])

    LOGS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    out = get_main_dfs(args.experiment, [args.model],
                       cache_only=args.cache_only)
    df, thresholds, display = out[args.model]

    n_by_dir = df.direction.value_counts().to_dict()
    print(f"\nDone: {display} | rows with parsed estimate: {len(df)} | "
          f"by condition: {n_by_dir}")
    print("Fresh thresholds:")
    for pk, t in sorted(thresholds.items()):
        print(f"  {pk}: {t}")
    print("\nUsage (this process):")
    total_cost = 0.0
    for mid, u in USAGE.items():
        print(f"  {mid}: {u['requests']} req, {u['in_tokens']:,} in, "
              f"{u['out_tokens']:,} out, ${u['cost_usd']:.2f}")
        total_cost += u["cost_usd"]
    print(f"  TOTAL measured cost: ${total_cost:.2f}")

    manifest = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "model_key": args.model, "experiment": args.experiment,
        "rows": len(df), "by_direction": n_by_dir,
        "thresholds": {k: v for k, v in thresholds.items()},
        "usage": USAGE, "wall_seconds": round(time.time() - t0, 1),
    }
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mpath = LOGS / f"{args.model}_{args.experiment}_{stamp}.json"
    mpath.write_text(json.dumps(manifest, indent=2, default=str))
    print(f"\nManifest: {mpath}")


if __name__ == "__main__":
    main()
