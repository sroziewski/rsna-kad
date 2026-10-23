#!/usr/bin/env python3
# RETIRED 2026-10-01: use scripts/export_kaggle.py --group/--group-weights/--prior-from (release/v7,
# commit dd39bfa). Kept only as the record of how main_v5_bal, main_v5_C and main_v6 were weighted.
"""Post-export edits of <export>/ensemble.json for the main_v5 probes (the exporter writes neither).

1. Weights. --balanced: members of one arm (run id minus its _s<seed>_full suffix) share 1/n_arms
   equally, i.e. seeds are averaged inside an arm and arms count equally (probe A). Otherwise every
   member gets the same weight (probe B). infer.py divides by the weight sum, so only ratios matter;
   they are normalised to sum to 1 for readability.
2. Prior. A full-data run's full/oof.parquet holds only the 58 gold rows, whose prevalence is
   enriched, and kaggle_export.prior_from_oof averages it into the fallback row that a failed study
   receives (ACL 0.376 instead of ~0.14). --prior-from replaces it with the prior of an ensemble.json
   built from 4,407-row OOF means of the same arm family.

  python qa/main_v5/finalize_ensemble.py export/<name> --prior-from export/main_v4/ensemble.json [--balanced]
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("bundle")
ap.add_argument("--prior-from", required=True)
ap.add_argument("--balanced", action="store_true")
a = ap.parse_args()

path = Path(a.bundle) / "ensemble.json"
doc = json.loads(path.read_text(encoding="utf-8"))


def arm(member: dict) -> str:
    return re.sub(r"_s\d+_full$", "", member["path"].split("/", 1)[1])


seeds = Counter(arm(m) for m in doc["members"])
for m in doc["members"]:
    m["weight"] = 1.0 / (len(seeds) * seeds[arm(m)]) if a.balanced else 1.0 / len(doc["members"])
doc["prior"] = json.loads(Path(a.prior_from).read_text(encoding="utf-8"))["prior"]
doc["prior_source"] = a.prior_from
path.write_text(json.dumps(doc, indent=1), encoding="utf-8")

for k, c in seeds.items():
    w = next(m["weight"] for m in doc["members"] if arm(m) == k)
    print(f"{k}: {c} seed(s), weight each {w:.4f}, arm total {w * c:.4f}")
print("members %d | weight sum %.6f | tta %s | prior %s" % (
    len(doc["members"]), sum(m["weight"] for m in doc["members"]),
    sorted({m["tta"] for m in doc["members"]}), [round(x, 3) for x in doc["prior"]]))
