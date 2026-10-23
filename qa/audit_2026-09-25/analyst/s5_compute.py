"""Q5 - GPU-hours per run from runs/*/fold*/train.log, and the Kaggle runtime budget.

GPU-busy = sum over epochs of the logged `train Xs eval Ys`; wall = last minus first timestamp of the
log (includes model load, restarts and waits). Writes s5_compute.md + s5_compute.csv."""
from __future__ import annotations

import json
import re
from datetime import datetime

import numpy as np
import pandas as pd

from common import OUT, PROJ

TS = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)")
EP = re.compile(r"epoch (\d+) done: .*?weak ([\d.]+).*?train (\d+)s eval (\d+)s (\d+) img/s")
FIN = re.compile(r"selected epoch (\d+) \((\w+)\) weak ([\d.]+).*best epoch (\d+) weak ([\d.]+)")
rows = []
for log in sorted(PROJ.glob("runs/*/fold*/train.log")):
    run, fold = log.parts[-3], log.parts[-2]
    ts, eps, fin = [], [], None
    for line in log.read_text(errors="replace").splitlines():
        m = TS.match(line)
        if m:
            ts.append(datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"))
        e = EP.search(line)
        if e:
            eps.append(tuple(float(x) for x in e.groups()))
        f = FIN.search(line)
        if f:
            fin = f.groups()
    if not ts:
        continue
    busy = sum(e[2] + e[3] for e in eps) / 3600
    rows.append({"run": run, "fold": fold, "epochs": len(eps), "gpu_busy_h": busy,
                 "wall_h": (ts[-1] - ts[0]).total_seconds() / 3600,
                 "img_s": float(np.median([e[4] for e in eps])) if eps else np.nan,
                 "finished": fin is not None,
                 "last_weak": eps[-1][1] if eps else np.nan,
                 "best_minus_last": (float(fin[4]) - float(fin[2])) if fin else np.nan})
D = pd.DataFrame(rows)
D.to_csv(OUT / "s5_compute.csv", index=False)
agg = D.groupby("run").agg(folds=("fold", "count"), finished=("finished", "sum"), epochs=("epochs", "sum"),
                           gpu_busy_h=("gpu_busy_h", "sum"), wall_h=("wall_h", "sum"), img_s=("img_s", "median"),
                           best_minus_last=("best_minus_last", "mean")).sort_values("gpu_busy_h", ascending=False)
L = ["# S5 compute - training GPU-hours from train.log", "",
     "| run | folds (finished) | epochs | GPU-busy h | log wall h | img/s | mean best-last weak (fold-level optimism) |",
     "|---|---|---|---|---|---|---|"]
for r, x in agg.iterrows():
    L.append(f"| {r} | {x.folds} ({int(x.finished)}) | {int(x.epochs)} | {x.gpu_busy_h:.1f} | {x.wall_h:.1f} | "
             f"{x.img_s:.0f} | {x.best_minus_last:+.4f} |")
L += ["", f"Total over all logged runs: GPU-busy {D.gpu_busy_h.sum():.1f} h, log wall {D.wall_h.sum():.1f} h "
      f"({len(D)} fold logs).", ""]

# Inference timings recorded by bench_infer.py (one A6000, 3 prep workers).
bench = {}
for tag, f in [("eff_v1 (E, K16, no TTA)", "qa/ensemble/time_eff_v1.json"),
               ("main_v2 (A+E, 10 members)", "qa/ensemble/time_main_v2.json"),
               ("main_v2 same-day", "qa/stage8b/time_main_v2_today.json"),
               ("main_v3 (A+E+C, 15 members)", "qa/stage8b/time_main_v3.json")]:
    d = json.loads((PROJ / f).read_text())
    best = min(d["runs"], key=lambda r: r["per_study_s"])
    bench[tag] = (d["per_study_s_best"], best["prep_s_mean"], best["gpu_s_mean"])
L += ["## Inference benchmarks (bench_infer.py, 1x A6000)", "",
      "| bundle | s/study | prep s (per worker) | GPU s | 1,300 studies A6000 min |", "|---|---|---|---|---|"]
for k, (s, p, g) in bench.items():
    L.append(f"| {k} | {s:.3f} | {p:.2f} | {g:.2f} | {s * 1300 / 60:.1f} |")
(OUT / "s5_compute.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))
