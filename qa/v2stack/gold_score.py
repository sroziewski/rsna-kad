"""Gold-58 scores of the v2stack e2e run: Raptor raw, heliosli compact, the public base, main_v5_C, and the
a-priori blend 0.7 * base + 0.3 * main_v5_C (W fixed 2026-10-11 before any gold number was seen).

Caveat: nartaa selected Raptor's checkpoint on gold-58 and heliosli trained with the 58 gold studies in its
pool config, so the base's gold score is optimistic and the blend-minus-base delta is biased against our leg.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/raid/sroziewski/sandbox/rsna_kad")
WK = ROOT / ".tmp/v2stack/e2e_full_work"
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "qa/full")]
import compare_pw as c  # noqa: E402
from rsna_kad.constants import DATA_DIR, ID_COL, LABEL_KEYS  # noqa: E402

gold = c.gold_frame(Path(DATA_DIR) / "train.csv")
gold[ID_COL] = gold[ID_COL].astype(str)
ids = sorted(gold[ID_COL])
y = gold.set_index(ID_COL).loc[ids, list(LABEL_KEYS)].to_numpy(np.float64) >= 0.5
cols = list(pd.read_csv(Path(DATA_DIR) / "sample_submission.csv", nrows=1).columns[1:])
assert len(cols) == len(LABEL_KEYS) == 12


def leg(name):
    df = pd.read_csv(WK / name, dtype={ID_COL: str}).set_index(ID_COL).loc[ids, cols]
    assert df.notna().all().all(), name
    return df.rank(pct=True)


cand = {"raptor": leg("raptor_raw.csv"), "compact": leg("ours_sub.csv"), "base": leg("_base.csv"),
        "main_v5_C": leg("_ours.csv"), "kernel": leg("submission.csv")}
for w in (0.2, 0.3, 0.4):
    cand[f"blend_{w:.1f}"] = (1 - w) * cand["base"] + w * cand["main_v5_C"]
d = (cand["kernel"] - cand["blend_0.3"].rank(pct=True)).abs().to_numpy().max()
print(f"kernel file vs offline blend_0.3 max rank diff {d:.4f}")
pl = {n: c.per_label(y, p.to_numpy()) for n, p in cand.items()}
print("gold-58 macro AUC")
for n in cand:
    print(f"  {n:10s} {np.nanmean(pl[n]):.4f}")
rng = np.random.default_rng(2026)
B = "blend_0.3"
for ref in ("base", "main_v5_C"):
    dd = np.empty(2000)
    a, b = cand[B].to_numpy(), cand[ref].to_numpy()
    for i in range(2000):
        ix = rng.integers(0, len(ids), len(ids))
        dd[i] = np.nanmean(c.per_label(y[ix], a[ix])) - np.nanmean(c.per_label(y[ix], b[ix]))
    print(f"{B} - {ref}: {np.nanmean(pl[B]) - np.nanmean(pl[ref]):+.4f} "
          f"[{np.percentile(dd, 2.5):+.4f}, {np.percentile(dd, 97.5):+.4f}] P={np.mean(dd > 0):.2f}")
print("label            raptor compact  base   ours  blend  rho(base,ours)")
for j, k in enumerate(cols):
    print(f"  {k:16s} {pl['raptor'][j]:.3f}  {pl['compact'][j]:.3f}  {pl['base'][j]:.3f}  {pl['main_v5_C'][j]:.3f}  "
          f"{pl[B][j]:.3f}  {cand['base'][k].corr(cand['main_v5_C'][k], method='spearman'):+.2f}")
