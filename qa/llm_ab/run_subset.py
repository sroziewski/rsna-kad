#!/usr/bin/env python3
"""Run one (model config, prompt variant) over the fixed A/B subset (.venv-llm, one GPU).

  qa/llm_ab/run_subset.py --config configs/llm_qwen3_32b_fp8.yaml --variant v1 \
      --tag fp8_base --out-dir data/labels

Reuses labels/llm.run_extractor unchanged (resumable jsonl, same schema, same guided decoding);
the only difference from scripts/extract_labels_llm.py is `ids=` from qa/llm_ab/eval_ids.txt and
the prompt-rule set coming from the config. Writes data/labels/llm_bf16_<tag>.{parquet,csv,
jsonl,stats.json,gold.json}. Nothing is written under $HOME (run it through scripts/run_screen.sh).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("HF_HOME", "/raid/sroziewski/cache/huggingface")
os.environ.pop("HF_HUB_CACHE", None)
os.environ.setdefault("VLLM_CACHE_ROOT", str(ROOT / ".vllm_cache"))
os.environ.setdefault("TRITON_CACHE_DIR", str(ROOT / ".triton_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".xdg_cache"))
os.environ.setdefault("TMPDIR", str(ROOT / ".tmp"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
os.environ.setdefault("DO_NOT_TRACK", "1")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
for _d in ("VLLM_CACHE_ROOT", "TRITON_CACHE_DIR", "XDG_CACHE_HOME", "TMPDIR"):
    Path(os.environ[_d]).mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from rsna_kad.constants import DATA_DIR, LABEL_KEYS  # noqa: E402
from rsna_kad.labels.llm import LLMConfig, run_extractor  # noqa: E402
from rsna_kad.labels.targets import gold_frame, gold_state_report  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--variant", default="v1", choices=["v1", "v2"])
    ap.add_argument("--tag", required=True, help="output name: data/labels/llm_bf16_<tag>.parquet")
    ap.add_argument("--ids", default=str(ROOT / "qa/llm_ab/eval_ids.txt"))
    ap.add_argument("--train", default=f"{DATA_DIR}/train.csv")
    ap.add_argument("--meta", default=str(ROOT / "analysis/out/train_meta.csv"))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--prompt-rules", default=None, help="override the config's prompt_rules")
    ap.add_argument("--retry-failed", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    os.chdir(ROOT)
    cfg = LLMConfig.load(args.config)
    if args.prompt_rules:
        cfg.prompt_rules = args.prompt_rules
    ids = [l.strip() for l in Path(args.ids).read_text(encoding="utf-8").splitlines() if l.strip()]
    out_dir = Path(args.out_dir or cfg.out_dir)
    out = out_dir / f"llm_bf16_{args.tag}.parquet"
    print(f"config {args.config} | model {cfg.model} | dtype {cfg.dtype} | quant {cfg.quantization} | "
          f"tp {cfg.tensor_parallel_size} | cpu_offload {cfg.cpu_offload_gb} GB | rules {cfg.prompt_rules} | "
          f"variant {args.variant} | {len(ids)} reports -> {out}")

    t0 = time.time()
    stats = run_extractor(args.train, cfg, args.variant, out, meta_csv=args.meta, ids=ids,
                          retry_failed=args.retry_failed)
    stats["tag"] = args.tag
    stats["prompt_rules"] = cfg.prompt_rules
    stats["n_ids_requested"] = len(ids)
    out.with_suffix(".stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in stats.items() if k != "config"}, indent=1))

    table = pd.read_parquet(out)
    rep = gold_state_report(table, gold_frame(args.train), n_boot=1000)
    out.with_suffix(".gold.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(f"gold n={rep['n']} macro {rep['macro_auc']:.4f} CI95 [{rep['ci95'][0]:.3f}, {rep['ci95'][1]:.3f}]")
    print("  " + " ".join(f"{k}={rep['per_label'][k]:.3f}" for k in LABEL_KEYS))
    print(f"total wall {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
