#!/usr/bin/env python3
"""Merged label table for targets v1.1: the stage-5 table plus a fourth voter.

  python scripts/make_labels_v1_1.py --voter qa/competitor/llm_labels/report_labels_v2.csv \
      --out data/labels/merged_v1_1.parquet

The fourth voter is either an external published table (`--voter <csv>`, PATH 2) or a second
local extractor's label table (`--voter <parquet>`, PATH 1). Either way it enters as `_v3`:
`data/labels/llm_v1.parquet` is read only and never rewritten, because training chains are
reading it (§4.1 tooling note).

Prints, and writes to <out>.voters.json, the numbers the §4.2 decision rule needs: gold-58 macro
AUC of the 3-voter table, of the new voter alone and of the 4-voter table, the paired bootstrap
delta over the 3-voter table, the per-label voter agreement, and the share of cells whose target
or weight actually moves.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rsna_kad.constants import DATA_DIR, ID_COL  # noqa: E402
from rsna_kad.labels.merge import add_voter, voter_agreement, voter_suffixes  # noqa: E402
from rsna_kad.labels.public import load_public_table, pick_low_yes_mapping  # noqa: E402
from rsna_kad.labels.schema import validate_label_table, write_label_table  # noqa: E402
from rsna_kad.labels.targets import gold_frame, label_scores  # noqa: E402
from rsna_kad.metrics import bootstrap_subset_auc, gold_subset, paired_delta, subset_auc  # noqa: E402


def load_voter(path: str | Path, lang_of: dict[str, str], gold: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """A published score/verdict CSV or one of our own label parquets."""
    path = Path(path)
    if path.suffix.lower() == ".csv" and "verdict" in pd.read_csv(path, nrows=1).columns.str.cat():
        info = pick_low_yes_mapping(path, gold, lang_of)
        table = load_public_table(path, lang_of, low_yes=str(info["chosen"]))
        info["kind"] = "public_csv"
        return table, info
    table = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    table[ID_COL] = table[ID_COL].astype(str)
    return table, {"kind": "label_table", "extractor": str(table["extractor"].iloc[0]),
                   "variant": str(table["variant"].iloc[0])}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="data/labels/llm_v1.parquet", help="the stage-5 merged table (read only)")
    ap.add_argument("--voter", required=True, help="the fourth voter: published CSV or a label table")
    ap.add_argument("--suffix", default="_v3")
    ap.add_argument("--train", default=f"{DATA_DIR}/train.csv")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="data/labels/merged_v1_1.parquet")
    args = ap.parse_args(argv)

    base = pd.read_parquet(args.base)
    base[ID_COL] = base[ID_COL].astype(str)
    gold = gold_frame(args.train)
    gold[ID_COL] = gold[ID_COL].astype(str)
    lang_of = dict(zip(base[ID_COL], base["lang"].astype(str)))

    voter, vinfo = load_voter(args.voter, lang_of, gold)
    merged = add_voter(base, voter, args.suffix)
    validate_label_table(merged, n_rows=len(base))

    before, after = voter_suffixes(base), voter_suffixes(merged)
    print(f"voters {before} -> {after}   ({vinfo})")

    # ---- gold-58: 3-voter vs new voter alone vs 4-voter, paired over studies
    cand = {"base_3voter": label_scores(base), "new_voter_alone": label_scores(voter),
            "merged_4voter": label_scores(merged)}
    for df in cand.values():
        df[ID_COL] = df[ID_COL].astype(str)
    common = set.intersection(*(set(df[ID_COL]) for df in cand.values()))
    g = gold[gold[ID_COL].isin(common)].reset_index(drop=True)
    gs = gold_subset(g)
    aligned = {n: gs.align(df) for n, df in cand.items()}
    scores = {n: subset_auc(gs, a) for n, a in aligned.items()}
    boots = bootstrap_subset_auc(gs, aligned, n_boot=args.n_boot, seed=args.seed)
    delta = paired_delta(boots["merged_4voter"], boots["base_3voter"])

    agree = voter_agreement(merged)
    covered = int(sum(1 for i in base[ID_COL] if i in set(voter[ID_COL].astype(str))))

    info = {
        "base": args.base, "voter": args.voter, "suffix": args.suffix, "voter_info": vinfo,
        "n_rows": int(len(merged)), "n_covered_by_new_voter": covered,
        "gold_n": int(len(g)),
        "gold_macro_auc": {n: s[0] for n, s in scores.items()},
        "gold_per_label": {n: s[1] for n, s in scores.items()},
        "paired_4voter_minus_3voter": delta,
        "voter_agreement": agree.to_dict("records"),
        "mean_unanimous_rate": float(agree["unanimous_rate"].mean()),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_label_table(merged, args.out)
    Path(str(args.out).replace(".parquet", ".voters.json")).write_text(
        json.dumps(info, indent=1, default=float), encoding="utf-8")

    print(f"\ngold-{info['gold_n']} macro AUC (state->score mapping, averaged over the voters present):")
    for n, s in scores.items():
        print(f"  {n:<16} {s[0]:.4f}")
    print(f"  4voter - 3voter: {delta['mean']:+.4f} [{delta['ci_lo']:+.4f}, {delta['ci_hi']:+.4f}] "
          f"P={delta['p_gt']:.3f}  (bar: >= 0.010 with P >= 0.95)")
    print(f"\nnew voter covers {covered}/{len(merged)} studies; "
          f"mean unanimity over {len(after)} voters {info['mean_unanimous_rate']:.3f}")
    print(agree.to_string(index=False))
    print(f"\nwrote {args.out} (+ .csv, .voters.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
