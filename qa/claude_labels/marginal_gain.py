"""Marginal value of the Claude voter on top of the voters we already have.

Compares rank-averaged voter sets on the gold rows of the 481-study subset, with a
paired bootstrap, so the question answered is "does adding Claude to {ours, gemma3,
public} help", not "does Claude help on its own".
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "qa" / "llm_ab"))
from voter_gain import load_voter, rank01  # noqa: E402

from rsna_kad.constants import DATA_DIR, ID_COL, LABEL_KEYS  # noqa: E402
from rsna_kad.labels.targets import gold_frame, label_scores  # noqa: E402
from rsna_kad.metrics import bootstrap_subset_auc, gold_subset, subset_auc  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TABLES = {
    "ours": ROOT / "data/labels/llm_v1.parquet",
    "gemma3": ROOT / "data/labels/llm_bf16_gemma3.parquet",
    "public": ROOT / "qa/competitor/llm_labels/report_labels_v2.csv",
    "claude": ROOT / "data/labels/claude_subset.parquet",
}


def rank_avg(scores: dict[str, pd.DataFrame], names: list[str], ids: list[str]) -> pd.DataFrame:
    out = pd.DataFrame({ID_COL: ids})
    for key in LABEL_KEYS:
        cols = [rank01(scores[n].set_index(ID_COL).loc[ids, f"{key}_pred"].to_numpy()) for n in names]
        out[f"{key}_pred"] = np.mean(cols, axis=0)
    return out


def main() -> None:
    base = pd.read_parquet(TABLES["ours"])
    base[ID_COL] = base[ID_COL].astype(str)
    lang_of = dict(zip(base[ID_COL], base["lang"].astype(str)))
    tables = {"ours": base}
    for name in ("gemma3", "public", "claude"):
        tables[name] = load_voter(str(TABLES[name]), lang_of)
    subset = {l.strip() for l in (ROOT / "qa/llm_ab/eval_ids.txt").read_text().splitlines() if l.strip()}
    ids = sorted(subset.intersection(*(set(t[ID_COL].astype(str)) for t in tables.values())))

    scores = {}
    for name, t in tables.items():
        t = t[t[ID_COL].astype(str).isin(set(ids))].reset_index(drop=True)
        s = label_scores(t)
        s[ID_COL] = s[ID_COL].astype(str)
        scores[name] = s

    cands = {}
    for k in (1, 2, 3, 4):
        for combo in itertools.combinations(["ours", "gemma3", "public", "claude"], k):
            if "ours" in combo:
                cands["+".join(combo)] = rank_avg(scores, list(combo), ids)

    gold = gold_frame(f"{DATA_DIR}/train.csv")
    gold[ID_COL] = gold[ID_COL].astype(str)
    gs = gold_subset(gold[gold[ID_COL].isin(set(ids))].reset_index(drop=True))
    aligned = {n: gs.align(df) for n, df in cands.items()}
    point = {n: subset_auc(gs, a)[0] for n, a in aligned.items()}
    boots = bootstrap_subset_auc(gs, aligned, n_boot=2000, seed=2026)

    def paired(a: str, b: str) -> tuple[float, float, float, float]:
        d = np.asarray(boots[a]) - np.asarray(boots[b])
        return point[a] - point[b], np.percentile(d, 2.5), np.percentile(d, 97.5), float((d > 0).mean())

    print(f"{len(ids)} studies")
    for n in sorted(point, key=point.get, reverse=True):
        print(f"  {n:28s} {point[n]:.4f}")
    rows = []
    for a, b in [("ours+gemma3+public+claude", "ours+gemma3+public"),
                 ("ours+gemma3+claude", "ours+gemma3"), ("ours+public+claude", "ours+public"),
                 ("ours+gemma3+public", "ours"), ("ours+claude", "ours")]:
        d, lo, hi, p = paired(a, b)
        rows.append({"with": a, "without": b, "delta": round(d, 4), "ci95": [round(lo, 4), round(hi, 4)],
                     "p_gt_0": round(p, 3), "clears_4_2": bool(d >= 0.010 and p >= 0.95)})
        print(f"  {a:28s} vs {b:22s} {d:+.4f} [{lo:+.4f},{hi:+.4f}] P={p:.3f}")
    (ROOT / "qa/claude_labels/marginal_gain.json").write_text(json.dumps({"point": point, "paired": rows}, indent=1))


if __name__ == "__main__":
    main()
