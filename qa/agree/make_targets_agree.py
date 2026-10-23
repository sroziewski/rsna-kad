#!/usr/bin/env python3
"""data/targets/v3_pseudo_full_agree: v3_pseudo_full with label cells the report labellers do not back masked out
(qa/agree, 2026-10-14; idea from forum 745861, Prateek Grover: train only where LLM labellers agree).

Voters per cell: llm_v1 (our Qwen merge), src_public (public table), llm_v1_gemma3 (Gemma-3-27B) and, for the six
findings it graded, glm_grade_v1.  Polarity: affirmed/yes = +, negated/no = -, hedged/borderline = 0, absent /
not_mentioned = no vote.  Rule, fixed before any model was trained:
  contested = some voter + and some voter -;
  weak      = not contested, at least one voter speaks, but fewer than two give a definite (+/-) call;
  contested or weak -> weight 0; every other cell keeps v3_pseudo_full's target and weight exactly.
Why: on gold-58, llm_v1's definite calls are wrong in 8/17 contested and 6/15 weak cells vs 96/467 elsewhere.
Rejected: Prateek's extra strength cut (> 0.9 / < 0.1) - our targets mix 0.5 teacher, so it would drop most rows.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/raid/sroziewski/sandbox/rsna_kad")
sys.path.insert(0, str(ROOT / "src"))
from rsna_kad.constants import ID_COL, LABEL_KEYS  # noqa: E402

V3, OUT = ROOT / "data/targets/v3_pseudo_full", ROOT / "data/targets/v3_pseudo_full_agree"
POL = {"affirmed": 1, "yes": 1, "negated": -1, "no": -1, "hedged": 0, "borderline": 0}


def masks():
    src = {n: pd.read_parquet(ROOT / f"data/labels/{n}.parquet").set_index(ID_COL)
           for n in ("llm_v1", "src_public", "llm_v1_gemma3")}
    glm = pd.read_parquet(ROOT / "data/labels/glm_grade_v1.parquet").set_index(ID_COL)
    out = {}
    for k in LABEL_KEYS:
        p = pd.DataFrame({n: d[f"{k}_state"].map(POL) for n, d in src.items()})
        if f"{k}_grade" in glm:
            p["glm"] = glm[f"{k}_grade"].map(POL).reindex(p.index)
        p = p.astype(float)
        cont = (p == 1).any(axis=1) & (p == -1).any(axis=1)
        weak = ~cont & (p.notna().sum(axis=1) > 0) & (p.isin([1, -1]).sum(axis=1) < 2)
        out[k] = cont | weak
    return pd.DataFrame(out)


if __name__ == "__main__":
    ref = pd.read_parquet(V3 / "full.parquet")
    m = masks().reindex(ref[ID_COL].astype(str).to_numpy())
    assert m.notna().all().all() and len(m) == len(ref) == 4349
    new = ref.copy()
    stats = {}
    for k in LABEL_KEYS:
        w = f"{k}_weight"
        hit = m[k].to_numpy(bool) & (ref[w].to_numpy() > 0)
        new.loc[hit, w] = ref[w].dtype.type(0)
        stats[k] = {"masked": int(hit.sum()), "of_weighted": int((ref[w] > 0).sum())}
        assert np.array_equal(new[f"{k}_target"].to_numpy(), ref[f"{k}_target"].to_numpy())
    tot = sum(s["masked"] for s in stats.values()) / sum(s["of_weighted"] for s in stats.values())
    OUT.mkdir(parents=True, exist_ok=True)
    new.to_parquet(OUT / "full.parquet", index=False)
    (OUT / "calibration_full.json").write_bytes((V3 / "calibration_full.json").read_bytes())
    man = json.loads((V3 / "manifest.json").read_text())
    man.update(version=OUT.name, derived_from=str(V3.relative_to(ROOT)), builder="qa/agree/make_targets_agree.py",
               rule=__doc__.split("Rule, fixed before any model was trained:")[1].split("Why:")[0].strip(),
               masked=stats, masked_share=tot)
    (OUT / "manifest.json").write_text(json.dumps(man, indent=1))
    for k, s in stats.items():
        print(f"  {k:10s} masked {s['masked']:4d} / {s['of_weighted']}")
    print(f"OK {OUT}: {tot:.1%} of weighted cells masked; targets identical to v3_pseudo_full")
