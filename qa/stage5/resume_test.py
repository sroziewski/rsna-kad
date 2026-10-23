"""Resumability of the LLM runner with a fake LLM (no vLLM, no GPU): interrupted run -> resume,
duplicate / torn jsonl lines, retry_failed. Runs entirely under .test_tmp."""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

ROOT = Path("/raid/sroziewski/sandbox/rsna_kad")
sys.path.insert(0, str(ROOT / "src"))
os.environ["TMPDIR"] = str(ROOT / ".test_tmp")
import rsna_kad.labels.llm as llm_mod  # noqa: E402
from rsna_kad.constants import LABEL_KEYS  # noqa: E402
from rsna_kad.labels.llm import LLMConfig, _read_jsonl, run_extractor  # noqa: E402

tmp = Path(tempfile.mkdtemp(dir=ROOT / ".test_tmp"))
ids = [f"1.2.{i}" for i in range(7)]
train = pd.DataFrame({"StudyInstanceUID": ids, "Report": [f"no effusion, ACL tear {i}" for i in range(7)],
                      **{c: [None] * 7 for c in ("ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture")}})
train.to_csv(tmp / "train.csv", index=False)
meta = pd.DataFrame({"StudyInstanceUID": ids, "lang": ["en"] * 7})
meta.to_csv(tmp / "meta.csv", index=False)


def good(i):
    return json.dumps({"knee_side": "left", "postoperative": False, "normal_template": False,
                       "labels": {k: {"state": "affirmed" if k == "acl" else "negated" if k == "effusion" else "absent",
                                      "severity": "unspecified" if k == "acl" else "none",
                                      "evidence": f"ACL tear {i}" if k == "acl" else "no effusion" if k == "effusion" else ""} for k in LABEL_KEYS}})


class FakeLLM:
    def __init__(self, fail_after_batches=None, truncate_ids=()):
        self.n = 0
        self.fail_after = fail_after_batches
        self.truncate_ids = set(truncate_ids)

    def chat(self, messages, params, **kw):
        self.n += 1
        if self.fail_after is not None and self.n > self.fail_after:
            raise KeyboardInterrupt("simulated crash")
        outs = []
        for msg in messages:
            rep = msg[-1]["content"]
            i = int(rep.split("ACL tear ")[1].split("\n")[0].strip().strip('"'))
            trunc = f"1.2.{i}" in self.truncate_ids
            text = good(i)[:40] if trunc else good(i)
            outs.append(SimpleNamespace(outputs=[SimpleNamespace(text=text, token_ids=[1, 2], finish_reason="length" if trunc else "stop")], prompt_token_ids=[1]))
        return outs


llm_mod.sampling_params = lambda cfg: None  # vLLM not importable in the shared venv
cfg = LLMConfig(batch_size=2, fewshot_dir=str(tmp / "nofewshot"), out_dir=str(tmp))
out = tmp / "llm_v1_t.parquet"

# 1) crash after the first batch
try:
    run_extractor(tmp / "train.csv", cfg, "v1", out, meta_csv=tmp / "meta.csv", llm=FakeLLM(fail_after_batches=1, truncate_ids=["1.2.1"]))
except KeyboardInterrupt:
    pass
done = _read_jsonl(out.with_suffix(".jsonl"))
print("after crash: jsonl records", len(done), sorted(done), "parquet exists:", out.exists())
# 2) torn last line + duplicate id line appended
with open(out.with_suffix(".jsonl"), "a", encoding="utf-8") as fh:
    fh.write(json.dumps({"StudyInstanceUID": "1.2.0", "lang": "en", "variant": "v1", "raw": good(0), "parse_ok": True, "prompt_tokens": 1, "gen_tokens": 2, "finish": "stop"}) + "\n")
    fh.write('{"StudyInstanceUID": "1.2.5", "lang": "en", "raw": "trunc')  # torn
# 3) resume
st = run_extractor(tmp / "train.csv", cfg, "v1", out, meta_csv=tmp / "meta.csv", llm=FakeLLM())
t = pd.read_parquet(out)
lines = [l for l in out.with_suffix(".jsonl").read_text().split("\n") if l.strip()]
recs, bad_lines = [], []
for l in lines:
    try:
        recs.append(json.loads(l))
    except ValueError:
        bad_lines.append(l[:80])
print("resume: n_resumed", st["n_resumed"], "n_run", st["n_run"], "table rows", len(t), "unique", t.StudyInstanceUID.nunique(), "ids==train", set(t.StudyInstanceUID) == set(ids),
      "parse_ok", int(t.parse_ok.sum()), "jsonl lines", len(lines), "parseable", len(recs), "unparseable", bad_lines, "jsonl unique ids", len({r["StudyInstanceUID"] for r in recs}), "truncated stat", st["truncated"])
print("  ids missing from jsonl after resume (would be re-run on a later resume):", sorted(set(ids) - {r["StudyInstanceUID"] for r in recs}))
print("  row 1.2.1 (truncated in first pass) parse_ok:", bool(t.set_index("StudyInstanceUID").loc["1.2.1", "parse_ok"]), "acl state:", str(t.set_index("StudyInstanceUID").loc["1.2.1", "acl_state"]))
# 4) retry_failed re-runs only the truncated one
st = run_extractor(tmp / "train.csv", cfg, "v1", out, meta_csv=tmp / "meta.csv", llm=FakeLLM(), retry_failed=True)
t = pd.read_parquet(out)
print("retry_failed: n_resumed", st["n_resumed"], "n_run", st["n_run"], "rows", len(t), "parse_ok", int(t.parse_ok.sum()), "truncated", st["truncated"],
      "acl 1.2.1:", str(t.set_index("StudyInstanceUID").loc["1.2.1", "acl_state"]), "evidence_ok_rate", round(st["evidence_ok_rate"], 3))
# 5) second resume with nothing to do: no LLM call
st = run_extractor(tmp / "train.csv", cfg, "v1", out, meta_csv=tmp / "meta.csv", llm=None)
print("no-op resume: n_resumed", st["n_resumed"], "n_run", st.get("n_run"), "rows", st["n_rows"])
# 6) merge with a parse-failed V2 row: does the failed voter count as disagreement?
from rsna_kad.labels.merge import merge_votes  # noqa: E402
from rsna_kad.labels.schema import label_row  # noqa: E402
from rsna_kad.labels.targets import TargetsConfig, calibrate, make_targets  # noqa: E402
v1 = t.copy()
v2 = t.copy()
v2.loc[v2.StudyInstanceUID == "1.2.3", "parse_ok"] = False
for k in LABEL_KEYS:
    v2.loc[v2.StudyInstanceUID == "1.2.3", f"{k}_state"] = "absent"
    v2.loc[v2.StudyInstanceUID == "1.2.3", f"{k}_severity"] = "none"
rules = pd.DataFrame([label_row(i, "en", "rules", "v0", None) for i in ids])
rules["parse_ok"] = True
mg = merge_votes(v1, v2, rules)
tc = TargetsConfig.load(ROOT / "configs/targets_v1.yaml")
tg = make_targets(mg, calibrate(mg, None, None, tc), tc).set_index("StudyInstanceUID")
r = mg.set_index("StudyInstanceUID").loc["1.2.3"]
print("V2 parse-failed row: merged parse_ok", bool(r.parse_ok), "acl_state", str(r.acl_state), "acl_state_v2", str(r.acl_state_v2), "acl_agree", bool(r.acl_agree),
      "-> acl target", float(tg.loc["1.2.3", "acl_target"]), "weight", float(tg.loc["1.2.3", "acl_weight"]), "| clean row 1.2.2: target", float(tg.loc["1.2.2", "acl_target"]), "weight", float(tg.loc["1.2.2", "acl_weight"]))
import shutil  # noqa: E402
shutil.rmtree(tmp)
