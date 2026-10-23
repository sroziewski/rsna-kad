"""Dump 25 random reports (stratified by language) with V1/V2/rules states for manual reading,
plus a sample of evidence_ok=False cells, plus the V2-parse-failure effect on targets."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/raid/sroziewski/sandbox/rsna_kad")
sys.path.insert(0, str(ROOT / "src"))
from rsna_kad.constants import LABEL_KEYS, LABELS  # noqa: E402
from rsna_kad.labels.merge import evidence_ratio  # noqa: E402

train = pd.read_csv("/raid/sroziewski/data/rsna_kad/train.csv")
reports = dict(zip(train.StudyInstanceUID, train.Report.fillna("").astype(str)))
gold_ids = set(train.loc[train[LABELS[0]].notna(), "StudyInstanceUID"])
m = pd.read_parquet(ROOT / "data/labels/llm_v1.parquet").set_index("StudyInstanceUID")
quota = {"en": 6, "es": 4, "tr": 3, "de": 3, "nl": 2, "fr": 2, "hr": 2, "el": 2, "bg": 1}
rng = np.random.default_rng(20260918)
chosen = []
for lang, n in quota.items():
    pool = sorted(i for i in m.index[(m.lang.astype(str) == lang)] if i not in gold_ids)
    chosen += list(rng.choice(pool, n, replace=False))
out = []
for sid in chosen:
    r = m.loc[sid]
    out.append(f"\n########## {sid}  lang={r.lang} parse_ok={r.parse_ok} knee_side={r.knee_side} postop={r.postoperative} normal_template={r.normal_template}")
    out.append("REPORT: " + reports[sid].replace("\r", "").strip())
    for k in LABEL_KEYS:
        out.append(f"  {k:9s} V1={str(r[f'{k}_state']):8s}/{str(r[f'{k}_severity']):11s} V2={str(r[f'{k}_state_v2']):8s}/{str(r[f'{k}_severity_v2']):11s} rules={str(r[f'{k}_rules_state']):8s} ev_ok={bool(r[f'{k}_evidence_ok'])!s:5s} | {str(r[f'{k}_evidence'])[:110]}")
Path(ROOT / "qa/stage5/spotcheck.txt").write_text("\n".join(out), encoding="utf-8")
print("wrote", len(chosen), "reports")

# evidence_ok False examples: ratio distribution + 15 examples
rows = []
for k in LABEL_KEYS:
    bad = m.index[(~m[f"{k}_evidence_ok"].to_numpy(bool)) & (m[f"{k}_state"].astype(str) != "absent")]
    for sid in bad:
        rows.append((sid, k, str(m.loc[sid, f"{k}_state"]), str(m.loc[sid, f"{k}_evidence"]), evidence_ratio(str(m.loc[sid, f"{k}_evidence"]), reports[sid]),
                     bool(m.loc[sid, f"{k}_state_v2"] != "absent") and evidence_ratio(str(m.loc[sid, f"{k}_evidence_v2"]), reports[sid]) < 0.8))
ev = pd.DataFrame(rows, columns=["sid", "key", "state", "evidence", "ratio_primary", "v2_bad"])
print("merged evidence_ok=False non-absent cells:", len(ev), "of which primary quote ratio>=0.8 (flag comes from V2 quote):", int((ev.ratio_primary >= 0.8).sum()),
      "primary ratio dist:", np.percentile(ev.ratio_primary, [0, 10, 25, 50, 75, 90, 100]).round(2).tolist(), "empty primary quote:", int((ev.evidence == "").sum()))
lo = ev[ev.ratio_primary < 0.8].sample(15, random_state=1)
lines = ["\n########## evidence_ok=False examples (primary quote not found, ratio<0.8)"]
for r in lo.itertuples():
    rep = reports[r.sid]
    lines.append(f"{r.sid[-8:]} {r.key} {r.state} ratio={r.ratio_primary:.2f} QUOTE: {r.evidence!r}\n   REPORT[:600]: {rep[:600]!r}")
Path(ROOT / "qa/stage5/spotcheck.txt").open("a", encoding="utf-8").write("\n".join(lines))
print("primary-quote-not-found by state:", ev[ev.ratio_primary < 0.8].state.value_counts().to_dict(), "by key:", ev[ev.ratio_primary < 0.8].key.value_counts().to_dict())

# V2 parse failures: effect on targets
v2 = pd.read_parquet(ROOT / "data/labels/llm_v2_raw.parquet").set_index("StudyInstanceUID")
fail = list(v2.index[~v2.parse_ok.to_numpy(bool)])
t0 = pd.read_parquet(ROOT / "data/targets/v1/fold0.parquet").set_index("StudyInstanceUID")
print("V2 parse-failed ids gold?", [i in gold_ids for i in fail])
for sid in fail:
    r = m.loc[sid]
    st = {k: str(r[f"{k}_state"]) for k in LABEL_KEYS}
    print(f"  {sid[-8:]} lang={r.lang} V1 states {st}")
    print(f"     targets fold0: " + " ".join(f"{k}={float(t0.loc[sid, f'{k}_target']):.2f}/w{float(t0.loc[sid, f'{k}_weight']):.2f}" for k in LABEL_KEYS))
