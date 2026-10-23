#!/usr/bin/env python3
"""Checks of a release/v7 A+B dry-run bundle (see dryrun.sh). Prints one PASS/FAIL line per check
and exits 1 if any fails.

  PYTHONPATH=.worktrees/release_v7/src python qa/release_v7/check_dryrun.py export/_dryrun_v7 qa/release_v7/dryrun_<tag>
"""
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from rsna_kad import kaggle_export as ke
from rsna_kad.constants import ID_COL, LABELS
from rsna_kad.infer import members_from_ensemble

bundle, out = Path(sys.argv[1]), Path(sys.argv[2])
doc = json.loads((bundle / "ensemble.json").read_text())
fails = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not ok:
        fails.append(name)


a = [m for m in doc["members"] if m.get("group") == "A"]
b = [m for m in doc["members"] if m.get("group") == "B"]
check("members 6 A + 1 B", len(a) == 6 and len(b) == 1, f"{len(a)} A, {len(b)} B")
check("A weights 1/12 each", all(abs(m["weight"] - 1 / 12) < 1e-12 for m in a), str({round(m["weight"], 6) for m in a}))
check("B weight 0.5", abs(b[0]["weight"] - 0.5) < 1e-12, str(b[0]["weight"]))
check("groups 0.5/0.5", doc.get("groups") == {"A": 0.5, "B": 0.5}, str(doc.get("groups")))
check("no TTA", all(m["tta"] is False for m in doc["members"]))
cp = ke.run_prior("runs/main_armc_convnextv2_t_384_p")
check("prior = Cp 5-fold OOF mean", np.allclose(doc["prior"], cp), f"{[round(x, 3) for x in doc['prior']]} <- {doc['prior_source']}")

check("dataset licence 'other'", doc["license"] == "other", doc["license"])
md5 = hashlib.md5((bundle / "LICENSE-DINOv3.md").read_bytes()).hexdigest()
check("LICENSE-DINOv3.md verbatim", md5 == "712846e16e4d72d12ae1052fe7e00133", md5)
notice = (bundle / "NOTICE").read_text()
check("NOTICE names B and only B", b[0]["path"] in notice and not any(m["path"] in notice for m in a))
meta = json.loads((bundle / "dataset-metadata.json").read_text())
check("dataset-metadata licence/slug", meta["licenses"] == [{"name": "other"}] and meta["id"].endswith("dryrun-v7"),
      json.dumps(meta)[:160])

bj = json.loads((bundle / (b[0]["path"] + ".json")).read_text())
check("B member json: DINOv3 @288, per_label_attn",
      "dinov3" in bj["arch"]["backbone"] and bj["arch"]["img_px"] == 288
      and bj["config"]["model"].get("head") == "per_label_attn",
      f"{bj['arch']['backbone']} {bj['arch']['img_px']} {bj['config']['model'].get('head')}")
mem = {m.prefix.rsplit("/", 1)[1]: m for m in members_from_ensemble(bundle)}
spec_b = mem[b[0]["path"].split("/")[1]].spec
check("B input spec 288 px, K16, views 130/110/130/130/90",
      spec_b[0] == 288 and spec_b[2] == 16 and tuple(spec_b[4]) == (130.0, 110.0, 130.0, 130.0, 90.0), str(spec_b))
check("A input spec 384 px, K24", all(mem[m["path"].split("/")[1]].spec[0] == 384 and mem[m["path"].split("/")[1]].spec[2] == 24
                                      for m in a))

log = (out / "notebook_dryrun_v7.log").read_text()
check("notebook: 7 members, mode main", "7 members, mode main" in log)
check("notebook: every member loaded", len(re.findall(r"INFO loaded ", log)) == 7)
check("notebook: 0 prior rows", re.search(r"\| 0 prior rows", log) is not None)
check("notebook: no traceback", "Traceback" not in log)
t = re.search(r"inference [0-9.]+s = ([0-9.]+)s/study", log)
print(f"INFO  CPU fp32 time {t.group(1) if t else '?'} s/study (A+B)")


def sub(p: Path) -> pd.DataFrame:
    return pd.read_csv(p).sort_values(ID_COL).reset_index(drop=True)


s, sb = sub(out / "submission_dryrun_v7.csv"), sub(out / "submission_dryrun_v7_b.csv")
v5c = sub(Path("export/submission_local_main_v5_C.csv"))
x, xb, xa = (d[list(LABELS)].to_numpy(float) for d in (s, sb, v5c))
check("submission: 3 rows, 12 cols, no NaN, in [0,1]",
      len(s) == 3 and list(s.columns) == [ID_COL, *LABELS] and np.isfinite(x).all() and x.min() >= 0 and x.max() <= 1,
      f"range {x.min():.4f}..{x.max():.4f}")
check("same study ids as main_v5_C", list(s[ID_COL]) == list(v5c[ID_COL]) == list(sb[ID_COL]))
err = np.abs(x - (0.5 * xa + 0.5 * xb)).max()
check("merge rule: blend == 0.5 x main_v5_C + 0.5 x B-only", err < 2e-5, f"max abs error {err:.2e}")
print(f"INFO  blend vs main_v5_C: mean abs {np.abs(x - xa).mean():.4f}; B-only vs main_v5_C: mean abs "
      f"{np.abs(xb - xa).mean():.4f}, Spearman {pd.Series(xb.ravel()).corr(pd.Series(xa.ravel()), method='spearman'):.3f}")
print("RESULT", "PASS" if not fails else f"FAIL {fails}")
sys.exit(1 if fails else 0)
