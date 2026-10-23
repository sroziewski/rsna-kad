#!/usr/bin/env python3
"""Fixed evaluation subset for the precision/prompt A/B (qa/llm_ab/report.md).

  58 gold studies  +  400 non-gold studies sampled with seed 0
  +  the 25 reports the stage-5 tester audited by hand (qa/stage5/spotcheck.py), so the audited
     error patterns can be counted on exactly the same reports as in qa/stage5/report.md.

Writes qa/llm_ab/eval_ids.txt (one StudyInstanceUID per line, sorted) and eval_subset.json with
the provenance of every id. Deterministic: rerunning it must not change the file.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rsna_kad.constants import DATA_DIR, LABELS  # noqa: E402

# qa/stage5/spotcheck.py, verbatim - the tester's hand-audited sample.
AUDIT_QUOTA = {"en": 6, "es": 4, "tr": 3, "de": 3, "nl": 2, "fr": 2, "hr": 2, "el": 2, "bg": 1}
AUDIT_SEED = 20260918


def audit_ids(labels: pd.DataFrame, gold_ids: set[str]) -> list[str]:
    m = labels.set_index("StudyInstanceUID")
    rng = np.random.default_rng(AUDIT_SEED)
    chosen: list[str] = []
    for lang, n in AUDIT_QUOTA.items():
        pool = sorted(i for i in m.index[(m.lang.astype(str) == lang)] if i not in gold_ids)
        chosen += list(rng.choice(pool, n, replace=False))
    return chosen


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=f"{DATA_DIR}/train.csv")
    ap.add_argument("--labels", default=str(ROOT / "data/labels/llm_v1.parquet"),
                    help="merged stage-5 table; only its id/lang columns are used")
    ap.add_argument("--n-sample", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "qa/llm_ab/eval_ids.txt"))
    args = ap.parse_args(argv)

    train = pd.read_csv(args.train)
    gold = sorted(train.loc[train[LABELS[0]].notna(), "StudyInstanceUID"])
    gold_set = set(gold)
    labels = pd.read_parquet(args.labels, columns=["StudyInstanceUID", "lang"])

    non_gold = sorted(set(train["StudyInstanceUID"]) - gold_set)
    rng = np.random.default_rng(args.seed)
    sample = sorted(rng.choice(non_gold, args.n_sample, replace=False).tolist())

    audit = audit_ids(labels, gold_set)
    ids = sorted(gold_set | set(sample) | set(audit))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(ids) + "\n", encoding="utf-8")
    meta = {"n_total": len(ids), "n_gold": len(gold), "n_sample": len(sample),
            "n_audit": len(audit), "n_audit_new": len(set(audit) - set(sample)),
            "sample_seed": args.seed, "audit_seed": AUDIT_SEED,
            "gold": gold, "sample": sample, "audit": sorted(audit)}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    lang = labels.set_index("StudyInstanceUID").loc[ids, "lang"].astype(str).value_counts().to_dict()
    print(f"{len(ids)} ids -> {out}  (gold {len(gold)}, seed-{args.seed} sample {len(sample)}, "
          f"audit {len(audit)} of which {meta['n_audit_new']} new)")
    print("languages:", lang)
    return 0


if __name__ == "__main__":
    sys.exit(main())
