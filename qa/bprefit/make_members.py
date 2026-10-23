"""bprefit step 2: the qoof_driver.py outputs as member files in Qile's format (StudyInstanceUID, set = oof | gold58, 12
findings), so Andrey's fit_stacker5.py runs on them unchanged (--qile-dir). OL-plane is merged as bpblur.py plane() does:
c = c.fillna(s); OLP = w * s + (1 - w) * c. A study with no sagittal/axial series (s missing) is NaN there and the kernel
ranks it 0.5; fit_stacker5 refuses NaN, so such rows take the coronal prediction (--fill-c, counted and printed).

  python -I make_members.py --work WORK --weights OL-plane_merge_weights.json --labels LABELS.csv --out DIR
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ID = "StudyInstanceUID"
NAMES = {"MTR": "member_mt_r50.csv", "MTH": "member_mt_hires.csv", "OLP": "member_ol_plane.csv"}


def read(p):
    return pd.read_csv(p, dtype={ID: str}).set_index(ID)


def merge_olp(s, c, wp, labels):
    c = c.fillna(s)
    olp = pd.DataFrame({f: wp[f] * s[f] + (1 - wp[f]) * c[f] for f in labels}, index=s.index.union(c.index))
    miss = olp.isna().any(axis=1)
    olp[miss] = c.reindex(olp.index)[miss]
    return olp, int(miss.sum())


def members(work, wp, labels, n_folds=5):
    w = Path(work)
    out, report = {}, {}
    for tag, setname in [(f"oof_f{k}", "oof") for k in range(n_folds)] + [("gold", "gold58")]:
        raw = {m: read(w / f"{tag}_{m}_foldmean.csv")[labels] for m in ("MTR", "MTH", "OLPS", "OLPC")}
        ids = raw["MTR"].index
        assert raw["MTH"].index.equals(ids), tag
        olp, filled = merge_olp(raw["OLPS"].reindex(ids), raw["OLPC"].reindex(ids), wp, labels)
        report[tag] = dict(rows=len(ids), olp_coronal_only=filled)
        for m, df in (("MTR", raw["MTR"]), ("MTH", raw["MTH"]), ("OLP", olp.loc[ids])):
            out.setdefault(m, []).append(df.assign(set=setname))
    res = {}
    for m, parts in out.items():
        df = pd.concat(parts)
        assert df.index.is_unique and df[labels].notna().all().all(), m
        res[m] = df[["set"] + labels]
    return res, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--labels", required=True, help="any csv with the 12 finding columns in submission order")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    labels = [c for c in pd.read_csv(a.labels, nrows=0).columns if c != ID]
    assert len(labels) == 12, labels
    wp = json.load(open(a.weights))["sagittal_weight"]
    res, report = members(a.work, wp, labels)
    for m, df in res.items():
        n = df.set.value_counts().to_dict()
        if n != {"oof": 4349, "gold58": 58}:
            sys.exit(f"{m}: unexpected rows {n}")
    d = Path(a.out)
    d.mkdir(parents=True, exist_ok=True)
    for m, df in res.items():
        df.to_csv(d / NAMES[m], index_label=ID)
    print("rows per tag", report)


if __name__ == "__main__":
    main()
