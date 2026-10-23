#!/usr/bin/env python3
"""Score the pre-declared TTA averages against the plain view on gold-58, paired (qa/tta2).

  python qa/tta2/compare.py

Averages (equal weights on probabilities, no fitting): base+crop; base+rotp5+rotm5;
base+crop+rotp5+rotm5. Paired bootstrap over the 58 gold studies (2000 replicates, the same
resample for every candidate). Adopt a TTA only if its gold-58 delta vs base is >= +0.002 (the bar
that dropped z-TTA). Inference cost = number of views x the plain pass, plus the measured wall time
of each view from timing.json (a rotated view also pays a CPU warp).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rsna_kad.constants import DATA_DIR, ID_COL, LABEL_KEYS  # noqa: E402
from rsna_kad.labels.targets import gold_frame  # noqa: E402
from rsna_kad.metrics import _rank_auc  # noqa: E402

QA = ROOT / "qa/tta2"
AVERAGES = {"base+crop": ("base", "crop"), "base+rot": ("base", "rotp5", "rotm5"),
            "base+crop+rot": ("base", "crop", "rotp5", "rotm5")}
N_BOOT, BAR = 2000, 0.002
PCOLS = [f"{k}_pred" for k in LABEL_KEYS]


def per_label(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    return np.array([_rank_auc(y[:, j], p[:, j]) for j in range(y.shape[1])])


def main() -> int:
    views = sorted({v for vs in AVERAGES.values() for v in vs})
    present = [v for v in views if (QA / v / "oof.parquet").exists()]
    gold = gold_frame(Path(DATA_DIR) / "train.csv")
    gold[ID_COL] = gold[ID_COL].astype(str)
    ids = sorted(gold[ID_COL])
    y = gold.set_index(ID_COL).loc[ids, list(LABEL_KEYS)].to_numpy(np.float64) >= 0.5
    pv = {}
    for v in present:
        o = pd.read_parquet(QA / v / "oof.parquet")
        o[ID_COL] = o[ID_COL].astype(str)
        pv[v] = o.set_index(ID_COL).loc[ids, PCOLS].to_numpy(np.float64)
    cand = {"base": pv["base"]}
    cand.update({f"view:{v}": pv[v] for v in present if v != "base"})
    cand.update({n: np.mean([pv[v] for v in vs], axis=0) for n, vs in AVERAGES.items() if set(vs) <= set(present)})

    rng = np.random.default_rng(2026)
    boots = {n: np.empty(N_BOOT) for n in cand}
    for b in range(N_BOOT):
        ix = rng.integers(0, len(ids), len(ids))
        for n, p in cand.items():
            boots[n][b] = np.nanmean(per_label(y[ix], p[ix]))
    timing = json.loads((QA / "timing.json").read_text()) if (QA / "timing.json").exists() else {}
    base_s = timing.get("base", {}).get("total_s")
    out = {"n_gold": len(ids), "views_present": present, "candidates": {}}
    base_pl = per_label(y, cand["base"])
    for n, p in cand.items():
        pl = per_label(y, p)
        d = boots[n] - boots["base"]
        vs = AVERAGES.get(n, (n.split(":")[-1],))
        secs = [timing.get(v, {}).get("total_s") for v in vs]
        out["candidates"][n] = {
            "macro": float(np.nanmean(pl)), "delta_vs_base": float(np.nanmean(pl) - np.nanmean(base_pl)),
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], "p_gt0": float((d > 0).mean()),
            "n_views": len(vs),
            "measured_cost_x": (float(sum(secs) / base_s) if base_s and all(secs) else None),
            "per_label_delta": dict(zip(LABEL_KEYS, map(float, pl - base_pl))),
            "adopt": bool(n in AVERAGES and np.nanmean(pl) - np.nanmean(base_pl) >= BAR)}
    (QA / "compare.json").write_text(json.dumps(out, indent=1))
    print(f"gold-58 ({len(ids)} studies), paired bootstrap {N_BOOT}; views present: {present}")
    for n, r in out["candidates"].items():
        cost = f"{r['n_views']} views" + (f", measured {r['measured_cost_x']:.2f}x" if r["measured_cost_x"] else "")
        print(f"  {n:16s} {r['macro']:.4f}  delta {r['delta_vs_base']:+.4f} [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]"
              f" P={r['p_gt0']:.2f}  {cost}{'  ADOPT' if r['adopt'] else ''}")
    for n in AVERAGES:
        if n in out["candidates"]:
            pl = out["candidates"][n]["per_label_delta"]
            print(f"  {n} per label: " + " ".join(f"{k} {v:+.3f}" for k, v in pl.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
