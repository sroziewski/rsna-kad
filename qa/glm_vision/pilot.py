"""GLM-5.3-Flash vision pilot (2026-10-18, user-requested, measurement only): can the internal VLM read the 12
findings straight from the images? If yes, it could relabel the external knees (MRNet/KneeMRI have no reports).

Per gold-58 study: one montage per plane (sag_fs, cor_fs or cor_t1, ax_fs; 12 evenly spaced slices from the central
80 %, 4x3 grid of 256 px from cache v1's 384 px volumes) + the competition's positive definitions; the model answers
a 0-100 probability per finding (guided JSON). Gold labels are used only for the scoring at the end.
Scored against main_v5_C on the same studies (the v3 e2e's _ours.csv): per-finding AUC, macro, and whether
0.9 rank(C) + 0.1 rank(GLM) adds anything. `python qa/glm_vision/pilot.py [n_studies]`
"""
from __future__ import annotations

import base64
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path("/raid/sroziewski/sandbox/rsna_kad")
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "qa/full")]
import compare_pw as c  # noqa: E402
from rsna_kad import cache  # noqa: E402
from rsna_kad.constants import DATA_DIR, ID_COL, LABEL_KEYS  # noqa: E402
from rsna_kad.labels.prompts import DEFINITIONS, THRESHOLDS  # noqa: E402

URL = os.environ["LLM_URL"]  # e.g. http://<llm-host>:8000/v1/chat/completions
MODEL = "opi-glm-4.6"  # = zai-org/GLM-5.3-Flash (GET /v1/models)
CACHE = "/raid/sroziewski/cache/rsna_kad/v1"
OUT = ROOT / "qa/glm_vision"
SLOTS = (("sagittal fat-suppressed", ("sag_fs", "sag_t1")), ("coronal", ("cor_fs", "cor_t1")),
         ("axial fat-suppressed", ("ax_fs", "ax_t1")))
SCHEMA = {"type": "object", "properties": {k: {"type": "integer", "minimum": 0, "maximum": 100} for k in LABEL_KEYS},
          "required": list(LABEL_KEYS), "additionalProperties": False}
PROMPT = ("You are a musculoskeletal radiologist. The images are one knee MRI study: one montage per plane, each "
          "showing 12 evenly spaced slices in order (left to right, top to bottom). For each finding give the "
          "probability (0-100) that it is present by this positive definition:\n"
          + "\n".join(f"- {k}: {DEFINITIONS[k]}; positive only if {THRESHOLDS[k]}" for k in LABEL_KEYS)
          + "\nAnswer with the JSON object only.")


def montage(vol: np.ndarray) -> bytes:
    n = len(vol)
    idx = np.linspace(0.1 * (n - 1), 0.9 * (n - 1), 12).round().astype(int)
    tiles = [cv2.resize(vol[i], (256, 256), interpolation=cv2.INTER_AREA) for i in idx]
    grid = np.vstack([np.hstack(tiles[r * 4:(r + 1) * 4]) for r in range(3)])
    return cv2.imencode(".png", grid)[1].tobytes()


def study_images(ix: pd.DataFrame, sid: str) -> list[tuple[str, bytes]]:
    rows = ix[(ix[ID_COL] == sid) & ix["selected"] & ix["ok"]]
    out = []
    for name, slots in SLOTS:
        for s in slots:
            r = rows[rows["slot"] == s]
            if len(r):
                r = r.sort_values("rank").iloc[0]
                out.append((name, montage(cache.load_series(CACHE, r["split"], sid, r["SeriesInstanceUID"]))))
                break
    return out


def ask(images: list[tuple[str, bytes]]) -> tuple[dict | None, float, str]:
    content = [{"type": "text", "text": PROMPT}]
    for name, png in images:
        content += [{"type": "text", "text": f"{name}:"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}}]
    body = {"model": MODEL, "temperature": 0, "max_tokens": 1500, "messages": [{"role": "user", "content": content}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "findings", "schema": SCHEMA}},
            "chat_template_kwargs": {"enable_thinking": False}}
    err = ""
    for _ in range(3):
        t0 = time.monotonic()
        try:
            req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                ch = json.loads(r.read())["choices"][0]
            if not ch["message"].get("content"):
                raise ValueError(f"empty content, finish_reason {ch.get('finish_reason')}")
            return json.loads(ch["message"]["content"]), time.monotonic() - t0, ""
        except Exception as e:  # noqa: BLE001 - retried, then recorded
            err = repr(e)[:200]
    return None, 0.0, err


def main() -> None:
    gold = c.gold_frame(Path(DATA_DIR) / "train.csv")
    gold[ID_COL] = gold[ID_COL].astype(str)
    ids = sorted(gold[ID_COL])[: int(sys.argv[1]) if len(sys.argv) > 1 else None]
    ix = cache.read_index(CACHE)
    ix = ix[ix[ID_COL].isin(ids)]
    t0 = time.monotonic()
    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(lambda s: (s, *ask(study_images(ix, s))), ids))
    wall = time.monotonic() - t0
    with open(OUT / "raw.jsonl", "w") as f:
        for s, ans, sec, err in res:
            f.write(json.dumps({ID_COL: s, "answer": ans, "s": round(sec, 1), "error": err}) + "\n")
    ok = [r for r in res if r[1] is not None]
    print(f"studies {len(ids)}, answered {len(ok)}, wall {wall:.0f}s, mean/request "
          f"{np.mean([r[2] for r in ok]):.1f}s")
    ids = [r[0] for r in ok]
    glm = np.array([[r[1][k] for k in LABEL_KEYS] for r in ok], np.float64)
    y = gold.set_index(ID_COL).loc[ids, list(LABEL_KEYS)].to_numpy(np.float64) >= 0.5
    cols = list(pd.read_csv(Path(DATA_DIR) / "sample_submission.csv", nrows=1).columns[1:])
    ours = pd.read_csv(ROOT / ".tmp/v2stack/v3_e2e_full_work/_ours.csv", dtype={ID_COL: str}).set_index(ID_COL)
    C = ours.loc[ids, cols].to_numpy(np.float64)
    rk = lambda a: pd.DataFrame(a).rank(pct=True).to_numpy()
    a_g, a_c, a_b = c.per_label(y, glm), c.per_label(y, C), c.per_label(y, 0.9 * rk(C) + 0.1 * rk(glm))
    lines = [f"GLM-5.3-Flash vision pilot, {len(ids)} gold studies ({len(res) - len(ok)} unanswered)",
             f"{'finding':18s} {'pos':>4s} {'GLM':>6s} {'C':>6s} {'.9C+.1GLM':>10s}  distinct GLM values"]
    for j, k in enumerate(LABEL_KEYS):
        lines.append(f"{cols[j]:18s} {int(y[:, j].sum()):4d} {a_g[j]:6.3f} {a_c[j]:6.3f} {a_b[j]:10.3f}  "
                     f"{len(np.unique(glm[:, j]))}")
    lines.append(f"{'MACRO':18s} {'':4s} {np.nanmean(a_g):6.3f} {np.nanmean(a_c):6.3f} {np.nanmean(a_b):10.3f}")
    lines.append(f"wall {wall:.0f}s for {len(ids)} studies, 6 parallel requests")
    (OUT / "pilot.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
