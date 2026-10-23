"""export/main_v5_C_ext: main_v5_C's bundle with its six students replaced by the three external-knee C seeds
(qa/ext, 2026-10-17 LB probe). Same layout as qa/oai/export_oai.py; members = the judge's seeds 2026/2126/2226 at 1/3
each, so the kernel's C leg is the "C ext 3 seeds" arm of qa/ext/gold_joint_ext.py."""
import datetime
import json
import shutil
from pathlib import Path

R = Path("/raid/sroziewski/sandbox/rsna_kad")
src, out = R / "export/main_v5_C", R / "export/main_v5_C_ext"
SEEDS = (2026, 2126, 2226)
assert not out.exists()
(out / "models").mkdir(parents=True)
shutil.copytree(src / "wheels", out / "wheels")
ens = json.loads((src / "ensemble.json").read_text())
proto = {k: v for k, v in ens["members"][0].items() if k not in ("path", "metrics")}
members, ids = [], []
for s in SEEDS:
    new = f"main_armc_convnextv2_t_384_p_full_ext_s{s}_full"
    run = R / "runs" / new[: -len("_full")] / "full"
    assert not (run / "last.pt").exists(), run
    shutil.copy2(run / "model.safetensors", out / "models" / f"{new}.safetensors")
    shutil.copy2(run / "bundle.json", out / "models" / f"{new}.json")
    met = json.loads((run / "metrics.json").read_text())
    members.append({**proto, "path": f"models/{new}", "weight": 1 / len(SEEDS),
                    "metrics": {k: met.get(k) for k in ("oof_weak_auc", "gold_auc", "epoch")}})
    ids.append(new)
ens["members"] = members
ens["version"] = (f"main_v5_C_ext {datetime.datetime.now().astimezone().isoformat(timespec='seconds')} "
                  + " ".join(ids))
(out / "ensemble.json").write_text(json.dumps(ens, indent=1))
(out / "VERSION").write_text(ens["version"] + "\n")
(out / "dataset-metadata.json").write_text(json.dumps({"title": "rsna-kad-weights-v5c-ext",
                                                       "id": "sroziews/rsna-kad-weights-v5c-ext",
                                                       "licenses": [{"name": "CC-BY-NC-4.0"}]}, indent=1))
print("ok", ids)
