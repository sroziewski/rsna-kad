#!/usr/bin/env python3
"""Stage 2: rule-voter label table (data/labels/rules_v0.parquet + csv) and its gold-58 report.

  python scripts/extract_labels_rules.py --train /raid/sroziewski/data/rsna_kad/train.csv \
      --meta analysis/out/train_meta.csv --out data/labels/rules_v0.parquet
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsna_kad.constants import DATA_DIR, LABEL_KEYS  # noqa: E402
from rsna_kad.labels.llm import load_reports  # noqa: E402
from rsna_kad.labels.rules import extract_rules_detail  # noqa: E402
from rsna_kad.labels.schema import label_row, validate_label_table, write_label_table  # noqa: E402
from rsna_kad.labels.schema import LabelCell, ReportLabels  # noqa: E402
from rsna_kad.labels.targets import gold_frame, gold_state_report  # noqa: E402


def build_rules_table(reports: pd.DataFrame, extractor: str = "rules", variant: str = "v0") -> pd.DataFrame:
    rows = []
    for r in reports.itertuples(index=False):
        votes, meta = extract_rules_detail(r.Report, r.lang)
        rec = ReportLabels(knee_side=str(meta["knee_side"]), postoperative=bool(meta["postoperative"]),
                           normal_template=bool(meta["normal_template"]),
                           labels={k: LabelCell(v.state, v.severity, v.evidence) for k, v in votes.items()})
        row = label_row(r.StudyInstanceUID, r.lang, extractor, variant, rec)
        for k, v in votes.items():
            row[f"{k}_mentioned"] = v.mentioned
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", default=f"{DATA_DIR}/train.csv")
    ap.add_argument("--meta", default="analysis/out/train_meta.csv", help="per-study language table (optional)")
    ap.add_argument("--out", default="data/labels/rules_v0.parquet")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args(argv)

    reports = load_reports(args.train, args.meta)
    table = build_rules_table(reports)
    validate_label_table(table, n_rows=len(reports))
    write_label_table(table, args.out)
    print(f"wrote {args.out}: {len(table)} rows")

    gold = gold_frame(args.train)
    rep = gold_state_report(table, gold, n_boot=args.n_boot)
    Path(args.out).with_suffix(".gold.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(f"gold-58 macro AUC {rep['macro_auc']:.4f}  CI95 [{rep['ci95'][0]:.3f}, {rep['ci95'][1]:.3f}]")
    for k in LABEL_KEYS:
        print(f"  {k:10s} {rep['per_label'][k]:.3f}")
    for lang, d in rep.get("per_lang", {}).items():
        print(f"  lang {lang}: n={d['n']} macro {d['macro_auc']:.3f}")
    # mention/affirmation rates per language: the cheapest vocabulary-coverage check
    summary = []
    for lang, sub in table.groupby("lang"):
        row = {"lang": lang, "n": len(sub)}
        for k in LABEL_KEYS:
            st = sub[f"{k}_state"].astype(str)
            row[f"{k}_aff"] = round(float((st == "affirmed").mean()), 3)
            row[f"{k}_neg"] = round(float((st == "negated").mean()), 3)
        summary.append(row)
    pd.DataFrame(summary).to_csv(Path(args.out).with_suffix(".rates.csv"), index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
