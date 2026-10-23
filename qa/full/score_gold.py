"""Gold-58 comparison of the seed ensemble and the full-data models against main_v4.

Fold runs contribute their out-of-fold predictions on the gold rows; full-data runs contribute the
gold predictions saved in <run>/full/oof.parquet. Every candidate is a mean of probabilities with
arms weighted equally (seeds are averaged inside an arm first), and all deltas are paired over the
same 58 studies.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from rsna_kad.constants import DATA_DIR, ID_COL, LABEL_KEYS, LABELS

ROOT = Path(__file__).resolve().parents[2]
FOLD_RUNS = {
    "Ap": "runs/main_convnext_s_384_e8_p", "Ep": "runs/eff_pseudo_256", "Cp": "runs/main_armc_convnextv2_t_384_p",
    "H2": "runs/main_armc_p_hpe2", "H3": "runs/main_armc_p_hpe3", "H4": "runs/main_armc_p_hpe4",
    "H5": "runs/main_armc_p_hpe5",
}
FULL_GLOBS = {"Afull": "main_convnext_s_384_e8_p_full_s*", "Efull": "eff_pseudo_256_full_s*",
              "Cfull": "main_armc_convnextv2_t_384_p_full_s*"}


def main() -> None:
    gold = pd.read_csv(f"{DATA_DIR}/train.csv")
    gold = gold[gold[LABELS[0]].notna()].assign(**{ID_COL: lambda d: d[ID_COL].astype(str)}).set_index(ID_COL)
    ids = list(gold.index)
    y = {k: gold[lab].to_numpy(int) for k, lab in zip(LABEL_KEYS, LABELS)}

    def load(path: Path) -> dict[str, np.ndarray]:
        o = pd.read_parquet(path).assign(**{ID_COL: lambda d: d[ID_COL].astype(str)}).set_index(ID_COL).loc[ids]
        return {k: o[f"{k}_pred"].to_numpy(float) for k in LABEL_KEYS}

    p = {n: load(ROOT / r / "oof.parquet") for n, r in FOLD_RUNS.items()}
    seeds = {}
    for arm, pat in FULL_GLOBS.items():
        runs = sorted(d for d in (ROOT / "runs").glob(pat) if (d / "full/oof.parquet").exists()
                      and not (d / "full/last.pt").exists())
        seeds[arm] = [d.name for d in runs]
        for d in runs:
            p[d.name] = load(d / "full/oof.parquet")

    def mean(parts):
        return {k: np.mean([q[k] for q in parts], axis=0) for k in LABEL_KEYS}

    arm_full = {a: mean([p[s] for s in ss]) for a, ss in seeds.items() if ss}
    c5 = mean([p[n] for n in ("Cp", "H2", "H3", "H4", "H5")])
    cands = {
        "main_v4 (Ap+Ep+Cp, fold OOF)": mean([p["Ap"], p["Ep"], p["Cp"]]),
        "Cp alone": p["Cp"],
        "C seed-ensemble (Cp+H2..H5)": c5,
        "main_v4 with C slot = seed-ensemble": mean([p["Ap"], p["Ep"], c5]),
    }
    for a, pa in arm_full.items():
        cands[f"{a} (mean of {len(seeds[a])} full seeds)"] = pa
    if len(arm_full) == 3:
        cands["main_v5 full-data (Afull+Efull+Cfull)"] = mean(list(arm_full.values()))
        cands["main_v5 full-data, C slot + seed-ensemble"] = mean(
            [arm_full["Afull"], arm_full["Efull"], mean([arm_full["Cfull"], c5])])

    def macro(q, rows):
        vals = [roc_auc_score(y[k][rows], q[k][rows]) for k in LABEL_KEYS if 0 < y[k][rows].sum() < len(rows)]
        return float(np.mean(vals))

    base = "main_v4 (Ap+Ep+Cp, fold OOF)"
    n = len(ids)
    boots = np.random.default_rng(2026).integers(0, n, (2000, n))
    full = np.arange(n)
    base_b = np.array([macro(cands[base], b) for b in boots])
    out = []
    print(f"seeds found: { {a: len(s) for a, s in seeds.items()} }")
    for c, q in cands.items():
        pt = macro(q, full)
        d = np.array([macro(q, b) for b in boots]) - base_b
        lo, hi, pr = np.percentile(d, 2.5), np.percentile(d, 97.5), float((d > 0).mean())
        out.append({"candidate": c, "gold": round(pt, 4), "delta": round(pt - macro(cands[base], full), 4),
                    "ci95": [round(lo, 4), round(hi, 4)], "p_gt_0": round(pr, 3)})
        print(f"{c:48s} gold {pt:.4f}  delta {pt - macro(cands[base], full):+.4f} [{lo:+.4f},{hi:+.4f}] P={pr:.3f}")
    (ROOT / "qa/full/score_gold.json").write_text(json.dumps({"seeds": seeds, "results": out}, indent=1))


if __name__ == "__main__":
    main()
