#!/usr/bin/env python3
"""Stage 5: LLM label extraction with vLLM (run with .venv-llm/bin/python, GPU 0 only).

  # zero-shot drafts for the few-shot files (unlabelled studies only)
  extract_labels_llm.py draft --ids data/fewshot/draft_ids.txt --out data/fewshot/drafts.json
  # one variant over all reports (resumable; results stream to data/labels/llm_<variant>.jsonl)
  extract_labels_llm.py run --variant v1
  extract_labels_llm.py run --variant v2
  # merge V1 + V2 + rules -> data/labels/llm_v1.parquet (+csv) and the gold-58 report
  extract_labels_llm.py merge --out data/labels/llm_v1.parquet

Environment defaults (set before torch/vLLM import): CUDA_VISIBLE_DEVICES=0, HF_HOME, caches under
the project dir; nothing is written under $HOME.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("HF_HOME", "/raid/sroziewski/cache/huggingface")
os.environ.pop("HF_HUB_CACHE", None)
os.environ.setdefault("VLLM_CACHE_ROOT", str(ROOT / ".vllm_cache"))
os.environ.setdefault("TRITON_CACHE_DIR", str(ROOT / ".triton_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".xdg_cache"))
os.environ.setdefault("TMPDIR", str(ROOT / ".test_tmp"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
# flashinfer's sampler JIT-compiles with nvcc, which the server does not have; greedy decoding does not need it
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
for _d in ("VLLM_CACHE_ROOT", "TRITON_CACHE_DIR", "XDG_CACHE_HOME", "TMPDIR"):
    Path(os.environ[_d]).mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from rsna_kad.constants import DATA_DIR, LABEL_KEYS  # noqa: E402
from rsna_kad.labels.llm import LLMConfig, build_llm, draft_fewshot, run_extractor  # noqa: E402
from rsna_kad.labels.merge import merge_summary, merge_votes  # noqa: E402
from rsna_kad.labels.schema import write_label_table  # noqa: E402
from rsna_kad.labels.targets import gold_frame, gold_state_report  # noqa: E402

log = logging.getLogger("extract_labels_llm")


def _print_gold(rep: dict, title: str) -> None:
    print(f"{title}: gold n={rep['n']} macro AUC {rep['macro_auc']:.4f} CI95 [{rep['ci95'][0]:.3f}, {rep['ci95'][1]:.3f}]")
    for k in LABEL_KEYS:
        print(f"  {k:10s} {rep['per_label'][k]:.3f}")
    for lang, d in rep.get("per_lang", {}).items():
        print(f"  lang {lang}: n={d['n']} macro {d['macro_auc']:.3f}")


def cmd_draft(args, cfg) -> int:
    ids = [l.strip() for l in Path(args.ids).read_text().splitlines() if l.strip() and not l.startswith("#")]
    drafts = draft_fewshot(args.train, cfg, ids, args.meta, variant=args.variant)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(drafts, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {len(drafts)} drafts to {args.out}")
    return 0


def cmd_run(args, cfg) -> int:
    variants = ["v1", "v2"] if args.variant == "both" else [args.variant]
    llm = build_llm(cfg)
    for variant in variants:
        out = Path(cfg.out_dir) / f"llm_{variant}_{args.tag}.parquet" if args.tag else Path(cfg.out_dir) / f"llm_{variant}.parquet"
        stats = run_extractor(args.train, cfg, variant, out, meta_csv=args.meta, limit=args.limit,
                              use_fewshot=not args.no_fewshot, llm=llm, retry_failed=args.retry_failed)
        print(json.dumps({k: v for k, v in stats.items() if k != "config"}, indent=1))
        if args.gold:
            table = pd.read_parquet(out)
            rep = gold_state_report(table, gold_frame(args.train), n_boot=200)
            _print_gold(rep, f"variant {variant}")
            Path(out).with_suffix(".gold.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    return 0


def cmd_merge(args, cfg) -> int:
    v1 = pd.read_parquet(args.v1 or Path(cfg.out_dir) / "llm_v1_raw.parquet")
    v2 = pd.read_parquet(args.v2 or Path(cfg.out_dir) / "llm_v2_raw.parquet")
    rules = pd.read_parquet(args.rules)
    merged = merge_votes(v1, v2, rules)
    write_label_table(merged, args.out)
    gold = gold_frame(args.train)
    rep = gold_state_report(merged, gold, n_boot=1000)
    rep_v1 = gold_state_report(v1, gold, n_boot=200)
    rep_v2 = gold_state_report(v2, gold, n_boot=200)
    rep_rules = gold_state_report(rules, gold, n_boot=200)
    summary = merge_summary(merged)
    gold_prev = {k: float(gold[k].mean()) for k in LABEL_KEYS}
    summary["gold_prevalence"] = summary["label"].map(gold_prev)
    stats = {"merged": rep, "v1": rep_v1, "v2": rep_v2, "rules": rep_rules,
             "summary": summary.to_dict(orient="records"),
             "parse_rate_v1": float(v1["parse_ok"].mean()), "parse_rate_v2": float(v2["parse_ok"].mean()),
             "n_rows": int(len(merged))}
    Path(args.out).with_suffix(".gold.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    summary.to_csv(Path(args.out).with_suffix(".summary.csv"), index=False)
    _print_gold(rep, "merged")
    print(f"v1 macro {rep_v1['macro_auc']:.4f}  v2 macro {rep_v2['macro_auc']:.4f}  rules macro {rep_rules['macro_auc']:.4f}")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs" / "llm_qwen3_32b_awq.yaml"))
    ap.add_argument("--train", default=f"{DATA_DIR}/train.csv")
    ap.add_argument("--meta", default=str(ROOT / "analysis" / "out" / "train_meta.csv"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("draft")
    d.add_argument("--ids", required=True)
    d.add_argument("--out", default=str(ROOT / "data" / "fewshot" / "drafts.json"))
    d.add_argument("--variant", default="v1")
    r = sub.add_parser("run")
    r.add_argument("--variant", choices=["v1", "v2", "both"], default="both")
    r.add_argument("--limit", type=int, default=None)
    r.add_argument("--tag", default="raw", help="output name suffix: llm_<variant>_<tag>.parquet")
    r.add_argument("--no-fewshot", action="store_true")
    r.add_argument("--gold", action="store_true", help="print the gold-58 report per variant")
    r.add_argument("--retry-failed", action="store_true", help="re-run studies whose result failed to parse or was truncated")
    m = sub.add_parser("merge")
    m.add_argument("--v1", default=None)
    m.add_argument("--v2", default=None)
    m.add_argument("--rules", default=str(ROOT / "data" / "labels" / "rules_v0.parquet"))
    m.add_argument("--out", default=str(ROOT / "data" / "labels" / "llm_v1.parquet"))
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    cfg = LLMConfig.load(args.config)
    os.chdir(ROOT)
    t0 = time.time()
    rc = {"draft": cmd_draft, "run": cmd_run, "merge": cmd_merge}[args.cmd](args, cfg)
    print(f"total wall {time.time() - t0:.0f}s")
    return rc


if __name__ == "__main__":
    sys.exit(main())
