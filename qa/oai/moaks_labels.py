"""Knee-level OAI baseline labels for PF OA, Lateral OA and Synovitis from the MOAKS tables, as re-exported (CC BY 4.0)
by 3DReasonKnee.  Every other competition finding stays masked.

The 3DReasonKnee export shifts the header by one column: `VERSION` holds READPRJ and `Unnamed: 3` holds VERSION.
MOAKS cartilage columns come in pairs per subregion: <col> = area grade (0-3), <col>.1 = full-thickness grade (0-3).
A knee read by several projects takes the maximum grade over its reads (reads differ by reader, not by visit).

Thresholds are chosen by prevalence: the competition's labels come from free-text reports, where "OA" is called on
much milder change than the MOAKS research definitions (gold-58 prevalence PF OA 0.36, Lateral OA 0.19, Synovitis
0.47), so `--report` prints prevalence per candidate rule and the rule is fixed before any model is trained.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

LBL = Path("/raid/sroziewski/data/rsna_kad_ext/3dreasonknee/data/labels")
PF = ["PM", "PL", "FMA", "FLA"]                      # patella medial/lateral facet, trochlea medial/lateral
LAT = ["FLC", "FLP", "TLA", "TLC", "TLP"]            # lateral femoral condyle central/posterior, lateral tibia
GOLD_PREV = {"PF OA": 0.36, "Lateral OA": 0.19, "Synovitis": 0.47}


def load():
    frames = []
    for f in ("taxonomymatch_kMRI_SQ_MOAKS_BICL00.csv", "taxonomymatch_kMRI_FNIH_SQ_MOAKS_BICL00.csv"):
        d = pd.read_csv(LBL / f, dtype=str)
        d = d.rename(columns={"VERSION": "READPRJ_", "READPRJ": "READER"})
        d["src"] = f.split("_")[1]
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    num = [c for c in d.columns if c.startswith("V00M")]
    d[num] = d[num].apply(pd.to_numeric, errors="coerce")
    d["SIDE"] = d["SIDE"].astype(int)
    return d.groupby(["ID", "SIDE"])[num].max(min_count=1).reset_index()


def cart(d, regions, full=False):
    cols = [f"V00MCM{r}" + (".1" if full else "") for r in regions]
    return d[cols].max(axis=1, skipna=True)


def rules(d):
    pf_a, pf_f, lat_a, lat_f = cart(d, PF), cart(d, PF, True), cart(d, LAT), cart(d, LAT, True)
    syn, eff = d["V00MSYIC"], d["V00MEFFWK"]
    r = {
        "PF OA": {"pf_area>=1": pf_a >= 1, "pf_area>=2": pf_a >= 2, "pf_area>=3": pf_a >= 3, "pf_full>=1": pf_f >= 1},
        "Lateral OA": {"lat_area>=1": lat_a >= 1, "lat_area>=2": lat_a >= 2, "lat_full>=1": lat_f >= 1,
                       "lat_area>=2|full>=1": (lat_a >= 2) | (lat_f >= 1)},
        "Synovitis": {"hoffa>=1": syn >= 1, "hoffa>=2": syn >= 2, "hoffa>=1|eff>=2": (syn >= 1) | (eff >= 2),
                      "hoffa>=2|eff>=2": (syn >= 2) | (eff >= 2)},
    }
    known = {"PF OA": pf_a.notna(), "Lateral OA": lat_a.notna(), "Synovitis": syn.notna()}
    return r, known


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    d = load()
    print("knees", len(d), "subjects", d["ID"].nunique())
    r, known = rules(d)
    for lab, cands in r.items():
        k = known[lab]
        print(f"{lab}: graded knees {int(k.sum())}, gold-58 prevalence {GOLD_PREV[lab]:.2f}")
        for name, m in cands.items():
            print(f"  {name:22s} prevalence {m[k].mean():.3f}")
