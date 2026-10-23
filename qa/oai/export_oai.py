"""export/main_v5_C_oai: main_v5_C's bundle with the six OAI-trained students swapped in (qa/oai, 2026-10-13).
Same layout as qa/stk/export_stk.py: member json = run bundle.json, member weights = run model.safetensors."""
import datetime
import json
import shutil
from pathlib import Path

R = Path("/raid/sroziewski/sandbox/rsna_kad")
src, out = R / "export/main_v5_C", R / "export/main_v5_C_oai"
assert not out.exists()
(out / "models").mkdir(parents=True)
shutil.copytree(src / "wheels", out / "wheels")
ens = json.loads((src / "ensemble.json").read_text())
ids = []
for m in ens["members"]:
    new = Path(m["path"]).name.replace("_p_full_s", "_p_full_oai_s")
    run = R / "runs" / new[: -len("_full")] / "full"
    assert not (run / "last.pt").exists(), run
    shutil.copy2(run / "model.safetensors", out / "models" / f"{new}.safetensors")
    shutil.copy2(run / "bundle.json", out / "models" / f"{new}.json")
    m["path"] = f"models/{new}"
    met = json.loads((run / "metrics.json").read_text())
    m["metrics"] = {"oof_weak_auc": met["oof_weak_auc"], "gold_auc": met["gold_auc"], "epoch": met["epoch"]}
    ids.append(new)
(out / "ensemble.json").write_text(json.dumps(ens, indent=1))
(out / "VERSION").write_text(f"main_v5_C_oai {datetime.datetime.now().astimezone().isoformat(timespec='seconds')} "
                             + " ".join(ids) + "\n")
(out / "dataset-metadata.json").write_text(json.dumps({"title": "rsna-kad-weights-v5c-oai",
                                                       "id": "sroziews/rsna-kad-weights-v5c-oai",
                                                       "licenses": [{"name": "CC-BY-NC-4.0"}]}, indent=1))
print("ok", ids)
