"""Threshold sensitivity for the confident-learning band + 10 hand-check cases with report text."""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from rsna_kad.constants import ID_COL, LABEL_KEYS, UNDER_REPORTED_KEYS  # noqa: E402

FOLDS = pd.read_csv("data/folds/folds_v1.csv")
FOLDS[ID_COL] = FOLDS[ID_COL].astype(str)


def targets_eval_table():
    parts = []
    for k in sorted(FOLDS["fold"].unique()):
        t = pd.read_parquet(f"data/targets/v1/fold{k}.parquet")
        t[ID_COL] = t[ID_COL].astype(str)
        parts.append(t[t[ID_COL].isin(set(FOLDS.loc[FOLDS["fold"] == k, ID_COL]))])
    return pd.concat(parts, ignore_index=True)


def load_oof(run, suf):
    o = pd.read_parquet(f"runs/{run}/oof.parquet")
    o[ID_COL] = o[ID_COL].astype(str)
    return o[[ID_COL] + [f"{k}_pred" for k in LABEL_KEYS]].rename(
        columns={f"{k}_pred": f"{k}_{suf}" for k in LABEL_KEYS})


tg = targets_eval_table()
lab = pd.read_parquet("data/labels/llm_v1.parquet")
lab[ID_COL] = lab[ID_COL].astype(str)
df = tg.merge(load_oof("main_convnext_s_384", "cn"), on=ID_COL).merge(
    load_oof("eff_effv2s_256", "ef"), on=ID_COL).merge(FOLDS[[ID_COL, "fold", "lang"]], on=ID_COL)
li = lab.set_index(ID_COL)

print("=== threshold sweep: cells where the mean OOF contradicts a high-weight LLM label")
print(" hi/lo  |  pos_contra  neg_contra  total   % of 52188 cells")
for hi, lo in [(0.9, 0.1), (0.85, 0.15), (0.8, 0.2), (0.75, 0.25), (0.7, 0.3), (0.6, 0.4)]:
    p = n = 0
    for k in LABEL_KEYS:
        t = df[f"{k}_target"].to_numpy(float)
        w = df[f"{k}_weight"].to_numpy(float)
        pm = (df[f"{k}_cn"].to_numpy(float) + df[f"{k}_ef"].to_numpy(float)) / 2.0
        p += int(((t >= hi) & (pm <= lo) & (w >= 0.5)).sum())
        n += int(((t <= lo) & (pm >= hi) & (w >= 0.5)).sum())
    print(f" {hi:.2f}/{lo:.2f} |  {p:10d}  {n:10d}  {p+n:5d}   {100.0*(p+n)/52188:.3f} %")

print("\n=== same, but on the STATE (affirmed vs negated) rather than the calibrated target")
print("    affirmed & OOF<=lo, or negated & OOF>=hi, weight>=0.5")
for hi, lo in [(0.9, 0.1), (0.8, 0.2), (0.7, 0.3)]:
    p = n = 0
    per = {}
    for k in LABEL_KEYS:
        st = li.loc[df[ID_COL], f"{k}_state"].astype(str).to_numpy()
        w = df[f"{k}_weight"].to_numpy(float)
        pm = (df[f"{k}_cn"].to_numpy(float) + df[f"{k}_ef"].to_numpy(float)) / 2.0
        a = int(((st == "affirmed") & (pm <= lo) & (w >= 0.5)).sum())
        b = int(((st == "negated") & (pm >= hi) & (w >= 0.5)).sum())
        per[k] = a + b
        p += a
        n += b
    print(f" {hi:.2f}/{lo:.2f} | affirmed-but-negative {p}  negated-but-positive {n}  total {p+n}  ({100.0*(p+n)/52188:.3f} %)")
    if hi == 0.8:
        print("   per key:", " ".join(f"{k}={v}" for k, v in per.items()))

# ------- 10 hand-check cases: the most extreme contradiction per key, up to 10, spread over keys
tr = pd.read_csv("/raid/sroziewski/data/rsna_kad/train.csv")[["StudyInstanceUID", "Report"]]
tr[ID_COL] = tr[ID_COL].astype(str)
rep = dict(zip(tr[ID_COL], tr["Report"]))

cases = []
for k in LABEL_KEYS:
    t = df[f"{k}_target"].to_numpy(float)
    w = df[f"{k}_weight"].to_numpy(float)
    pm = (df[f"{k}_cn"].to_numpy(float) + df[f"{k}_ef"].to_numpy(float)) / 2.0
    st = li.loc[df[ID_COL], f"{k}_state"].astype(str).to_numpy()
    sv = li.loc[df[ID_COL], f"{k}_severity"].astype(str).to_numpy()
    ev = li.loc[df[ID_COL], f"{k}_evidence"].astype(str).to_numpy()
    m = (((t >= 0.9) & (pm <= 0.1)) | ((t <= 0.1) & (pm >= 0.9))) & (w >= 0.5)
    idx = np.where(m)[0]
    if len(idx) == 0:
        continue
    # most extreme = largest |target - oof|
    idx = idx[np.argsort(-np.abs(t[idx] - pm[idx]))]
    for i in idx[:2]:
        cases.append({"key": k, "sid": df[ID_COL].iloc[i], "lang": df["lang"].iloc[i],
                      "fold": int(df["fold"].iloc[i]), "state": st[i], "sev": sv[i],
                      "evidence": ev[i], "target": float(t[i]), "w": float(w[i]),
                      "cn": float(df[f"{k}_cn"].iloc[i]), "ef": float(df[f"{k}_ef"].iloc[i]),
                      "gap": float(abs(t[i] - pm[i]))})
cases.sort(key=lambda c: -c["gap"])
seen, pick = set(), []
for c in cases:            # one per key first, then fill
    if c["key"] not in seen:
        seen.add(c["key"])
        pick.append(c)
for c in cases:
    if len(pick) >= 10:
        break
    if c not in pick:
        pick.append(c)
pick = pick[:10]

print("\n\n########## 10 HAND-CHECK CASES ##########")
for n, c in enumerate(pick, 1):
    print(f"\n--- [{n}] key={c['key']} study={c['sid']} lang={c['lang']} fold={c['fold']}")
    print(f"    LLM state={c['state']}/{c['sev']}  target={c['target']:.3f} w={c['w']:.2f}  "
          f"OOF convnext={c['cn']:.3f} eff={c['ef']:.3f}")
    print(f"    evidence quote: {c['evidence'][:160]}")
    print("    REPORT:")
    txt = str(rep.get(c["sid"], "<missing>")).replace("\r", " ")
    for line in txt.split("\n"):
        line = line.strip()
        if line:
            print("      " + line[:200])
