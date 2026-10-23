#!/usr/bin/env python3
"""Does a candidate fourth voter clear the §4.2 bar? Shared venv, CPU only.

  qa/llm_ab/voter_gain.py --base data/labels/llm_v1.parquet \
      --voter gemma3=data/labels/llm_bf16_gemma3.parquet \
      --voter public=qa/competitor/llm_labels/report_labels_v2.csv \
      --ids qa/llm_ab/eval_ids.txt --out qa/llm_ab/voter_gain

Reports, on the gold rows inside `--ids`, three numbers per candidate:

  alone        gold macro AUC of the candidate's own state->score mapping;
  rank_avg     gold macro AUC of the rank-average of (ours, candidate), ranked over the whole id
               set - the estimator qa/competitor/analysis.md §6 measured;
  merge_4v     gold macro AUC after the candidate enters the merged table as `_v3`, i.e. the mean
               of the mapped state scores - the estimator that actually produces targets v1.1.

plus the paired bootstrap of each against ours alone, over the *same* resampled studies, and the
per-label extractor correlation over the full id set (a voter that agrees with ours everywhere
adds nothing however good it is). Decision rule: accept at >= 0.010 with P >= 0.95.

`rank_avg` is reported because it is the published number; `merge_4v` is what a decision must be
taken on, because that is the table `scripts/make_targets.py` reads.
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

from rsna_kad.constants import DATA_DIR, ID_COL, LABEL_KEYS  # noqa: E402
from rsna_kad.labels.merge import add_voter  # noqa: E402
from rsna_kad.labels.public import load_public_table  # noqa: E402
from rsna_kad.labels.targets import gold_frame, label_scores  # noqa: E402
from rsna_kad.metrics import bootstrap_subset_auc, gold_subset, paired_delta, subset_auc  # noqa: E402


def rank01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < 2:
        return np.zeros(n)
    order = np.argsort(x, kind="mergesort")
    s = x[order]
    new = np.empty(n, dtype=bool)
    new[0] = True
    np.not_equal(s[1:], s[:-1], out=new[1:])
    group = np.cumsum(new) - 1
    starts = np.flatnonzero(new)
    ends = np.append(starts[1:], n) - 1
    r = np.empty(n, dtype=np.float64)
    r[order] = (starts[group] + ends[group]) / 2.0
    return r / (n - 1)


def rank_average(a: pd.DataFrame, b: pd.DataFrame, ids: list[str]) -> pd.DataFrame:
    ai, bi = a.set_index(ID_COL).loc[ids], b.set_index(ID_COL).loc[ids]
    out = pd.DataFrame({ID_COL: ids})
    for key in LABEL_KEYS:
        out[f"{key}_pred"] = (rank01(ai[f"{key}_pred"].to_numpy())
                              + rank01(bi[f"{key}_pred"].to_numpy())) / 2.0
    return out


def load_voter(path: str, lang_of: dict[str, str]) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".csv" and "verdict" in pd.read_csv(p, nrows=1).columns.str.cat():
        return load_public_table(p, lang_of, low_yes="hedged")
    t = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    t[ID_COL] = t[ID_COL].astype(str)
    return t


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=str(ROOT / "data/labels/llm_v1.parquet"))
    ap.add_argument("--voter", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--ids", default=str(ROOT / "qa/llm_ab/eval_ids.txt"))
    ap.add_argument("--train", default=f"{DATA_DIR}/train.csv")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=str(ROOT / "qa/llm_ab/voter_gain"))
    args = ap.parse_args(argv)

    base = pd.read_parquet(args.base)
    base[ID_COL] = base[ID_COL].astype(str)
    lang_of = dict(zip(base[ID_COL], base["lang"].astype(str)))
    gold = gold_frame(args.train)
    gold[ID_COL] = gold[ID_COL].astype(str)

    ids_all = [l.strip() for l in Path(args.ids).read_text(encoding="utf-8").splitlines() if l.strip()] \
        if args.ids else list(base[ID_COL])

    voters: dict[str, pd.DataFrame] = {}
    for spec in args.voter:
        name, _, path = spec.partition("=")
        voters[name] = load_voter(path, lang_of)

    # The id set every candidate can speak about, so all deltas are measured on the same studies.
    ids = sorted(set(ids_all) & set(base[ID_COL]) & set.intersection(
        *(set(v[ID_COL].astype(str)) for v in voters.values())))
    base_ids = base[base[ID_COL].isin(set(ids))].reset_index(drop=True)
    p_base = label_scores(base_ids)
    p_base[ID_COL] = p_base[ID_COL].astype(str)

    cand: dict[str, pd.DataFrame] = {"ours_alone": p_base}
    for name, v in voters.items():
        v_ids = v[v[ID_COL].astype(str).isin(set(ids))].reset_index(drop=True)
        p_v = label_scores(v_ids)
        p_v[ID_COL] = p_v[ID_COL].astype(str)
        cand[f"{name}_alone"] = p_v
        cand[f"{name}_rank_avg"] = rank_average(p_base, p_v, ids)
        merged = add_voter(base_ids, v_ids, "_v3")
        p_m = label_scores(merged)
        p_m[ID_COL] = p_m[ID_COL].astype(str)
        cand[f"{name}_merge_4v"] = p_m

    g = gold[gold[ID_COL].isin(set(ids))].reset_index(drop=True)
    gs = gold_subset(g)
    aligned = {n: gs.align(df) for n, df in cand.items()}
    scores = {n: subset_auc(gs, a) for n, a in aligned.items()}
    boots = bootstrap_subset_auc(gs, aligned, n_boot=args.n_boot, seed=args.seed)

    res: dict = {"n_ids": len(ids), "n_gold": int(len(g)), "base": args.base,
                 "macro": {n: s[0] for n, s in scores.items()},
                 "per_label": {n: s[1] for n, s in scores.items()}, "deltas": {}, "correlation": {}}
    for n in cand:
        if n != "ours_alone":
            res["deltas"][n] = paired_delta(boots[n], boots["ours_alone"])
    for name in voters:
        a = p_base.set_index(ID_COL).loc[ids]
        b = cand[f"{name}_alone"].set_index(ID_COL).loc[ids]
        c = {}
        for key in LABEL_KEYS:
            x, y = a[f"{key}_pred"].to_numpy(float), b[f"{key}_pred"].to_numpy(float)
            c[key] = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else float("nan")
        res["correlation"][name] = {"per_label": c, "mean": float(np.nanmean(list(c.values())))}
        for tag in ("parse_rate", "evidence_ok_rate"):
            v = voters[name]
            if tag == "parse_rate" and "parse_ok" in v.columns:
                res.setdefault("voter_stats", {}).setdefault(name, {})[tag] = float(v["parse_ok"].mean())

    rows = []
    for n, s in scores.items():
        d = res["deltas"].get(n)
        verdict = "-"
        if d:
            verdict = ("ACCEPT" if d["mean"] >= 0.010 and d["p_gt"] >= 0.95 else
                       "tie/bundle" if d["mean"] >= 0.005 and d["p_gt"] >= 0.95 else "reject")
        rows.append([n, f"{s[0]:.4f}",
                     f"{d['mean']:+.4f}" if d else "-",
                     f"[{d['ci_lo']:+.4f},{d['ci_hi']:+.4f}]" if d else "-",
                     f"{d['p_gt']:.3f}" if d else "-", verdict])
    hdr = ["candidate", f"gold-{res['n_gold']} AUC", "delta vs ours", "CI95", "P(>0)", "§4.2"]
    md = [f"# Fourth-voter gain on {len(ids)} studies ({res['n_gold']} gold)", "",
          "| " + " | ".join(hdr) + " |", "|" + "|".join(["---"] * len(hdr)) + "|"]
    md += ["| " + " | ".join(r) + " |" for r in rows]
    md += ["", "Per-label extractor correlation with ours (low = diverse voter):", ""]
    for name, c in res["correlation"].items():
        worst = sorted(c["per_label"].items(), key=lambda kv: kv[1])[:3]
        md.append(f"- **{name}** mean {c['mean']:.3f}; lowest " +
                  ", ".join(f"{k} {v:.3f}" for k, v in worst))
    md += ["", "Per-label gold AUC:", "",
           "| candidate | " + " | ".join(LABEL_KEYS) + " |",
           "|" + "|".join(["---"] * (len(LABEL_KEYS) + 1)) + "|"]
    for n, s in scores.items():
        md.append(f"| {n} | " + " | ".join(f"{s[1][k]:.3f}" for k in LABEL_KEYS) + " |")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"\nwrote {out.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
