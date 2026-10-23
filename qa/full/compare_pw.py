#!/usr/bin/env python3
"""Pairwise-ranking loss vs plain BCE, full-data C_p, gold-58 (pre-declared, coordinator 2026-09-30).

  python qa/full/compare_pw.py
  python qa/full/compare_pw.py --new main_armc_convnextv2_t_384_p_full_mri_s{} --bar 0.002 --p-bar 0       --out qa/mriaug/compare_mri.json      # MRI-aug stage 1 (stage-2 trigger: delta >= +0.002)

Primary: 3-seed mean of runs/main_armc_convnextv2_t_384_p_full_pw_s{2026,2126,2226}/full vs the 3-seed
mean of runs/main_armc_convnextv2_t_384_p_full_s{same seeds}/full, paired bootstrap over the 58 studies
(2000 replicates). Accept only at delta >= +0.004 with P(delta > 0) >= 0.95, else a tie. Also the three
per-seed pairs and per-label deltas. Refuses while any of the six runs is being written or unfinished.
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

from rsna_kad import runguard  # noqa: E402
from rsna_kad.constants import DATA_DIR, ID_COL, LABEL_KEYS  # noqa: E402
from rsna_kad.labels.targets import gold_frame  # noqa: E402
from rsna_kad.metrics import _rank_auc  # noqa: E402

SEEDS = (2026, 2126, 2226)
BASE = "main_armc_convnextv2_t_384_p_full_s{}"
PW = "main_armc_convnextv2_t_384_p_full_pw_s{}"
N_BOOT, BAR, P_BAR = 2000, 0.004, 0.95
PCOLS = [f"{k}_pred" for k in LABEL_KEYS]


def load(run: str, ids: list[str]) -> np.ndarray:
    o = pd.read_parquet(ROOT / "runs" / run / "full" / "oof.parquet")
    o[ID_COL] = o[ID_COL].astype(str)
    return o.set_index(ID_COL).loc[ids, PCOLS].to_numpy(np.float64)


def per_label(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    return np.array([_rank_auc(y[:, j], p[:, j]) for j in range(y.shape[1])])


def main(argv=None) -> int:
    global PW, SEEDS, BAR, P_BAR
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--new", default=PW, help="run-id pattern of the variant, {} = seed")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--bar", type=float, default=BAR)
    ap.add_argument("--p-bar", type=float, default=P_BAR)
    ap.add_argument("--out", default="qa/full/compare_pw.json")
    args = ap.parse_args(argv)
    PW, SEEDS, BAR, P_BAR = args.new, tuple(args.seeds), args.bar, args.p_bar
    runs = [ROOT / "runs" / f.format(s) for s in SEEDS for f in (BASE, PW)]
    problems = [e for rd in runs for e in runguard.export_problems([rd], range(5))]
    if problems:
        print("refusing:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 2
    gold = gold_frame(Path(DATA_DIR) / "train.csv")
    gold[ID_COL] = gold[ID_COL].astype(str)
    ids = sorted(gold[ID_COL])
    y = gold.set_index(ID_COL).loc[ids, list(LABEL_KEYS)].to_numpy(np.float64) >= 0.5
    cand = {f"base_s{s}": load(BASE.format(s), ids) for s in SEEDS}
    cand.update({f"new_s{s}": load(PW.format(s), ids) for s in SEEDS})
    cand["base_mean3"] = np.mean([cand[f"base_s{s}"] for s in SEEDS], axis=0)
    cand["new_mean3"] = np.mean([cand[f"new_s{s}"] for s in SEEDS], axis=0)
    pairs = [("new_mean3", "base_mean3")] + [(f"new_s{s}", f"base_s{s}") for s in SEEDS]

    rng = np.random.default_rng(2026)
    boots = {n: np.empty(N_BOOT) for n in cand}
    for b in range(N_BOOT):
        ix = rng.integers(0, len(ids), len(ids))
        for n, p in cand.items():
            boots[n][b] = np.nanmean(per_label(y[ix], p[ix]))
    out = {"n_gold": len(ids), "macro": {}, "pairs": {}, "per_label": {}}
    for n, p in cand.items():
        pl = per_label(y, p)
        out["macro"][n] = float(np.nanmean(pl))
        out["per_label"][n] = dict(zip(LABEL_KEYS, map(float, pl)))
    for a, b in pairs:
        d = boots[a] - boots[b]
        out["pairs"][f"{a} - {b}"] = {"delta": out["macro"][a] - out["macro"][b],
                                      "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                                      "p_gt0": float((d > 0).mean())}
    prim = out["pairs"]["new_mean3 - base_mean3"]
    out["verdict"] = ("ACCEPT" if prim["delta"] >= BAR and prim["p_gt0"] >= P_BAR else "TIE (not accepted)")
    out["rule"] = {"variant": PW, "base": BASE, "seeds": list(SEEDS), "bar": BAR, "p_bar": P_BAR}
    (ROOT / args.out).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / args.out).write_text(json.dumps(out, indent=1))

    print(f"gold-58 macro AUC ({len(ids)} studies, {N_BOOT} paired bootstrap replicates)")
    for n in cand:
        print(f"  {n:12s} {out['macro'][n]:.4f}")
    for k, v in out["pairs"].items():
        print(f"  {k:24s} {v['delta']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] P={v['p_gt0']:.2f}")
    print("per-label delta (new_mean3 - base_mean3):")
    for k in LABEL_KEYS:
        print(f"  {k:10s} {out['per_label']['base_mean3'][k]:.3f} -> {out['per_label']['new_mean3'][k]:.3f}"
              f" ({out['per_label']['new_mean3'][k] - out['per_label']['base_mean3'][k]:+.3f})")
    print(f"verdict (>= +{BAR} with P >= {P_BAR}): {out['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
