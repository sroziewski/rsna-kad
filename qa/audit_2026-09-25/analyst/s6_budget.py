"""Q5/Q6 - Kaggle runtime calibration, remaining 9 h budget and the efficiency-score arithmetic.

Inputs (all recorded elsewhere, nothing new measured on GPU):
  - bench_infer.py JSONs (1x A6000, 3 prep workers): eff_v1, main_v2 (same-day), main_v3
  - MVP hidden rerun 25 min for ~1,300 studies at 0.53 s/study bench (architecture §4.1, stage 4)
  - main_v2: submitted 2026-09-21 14:44:43 UTC (kaggle submissions list), first seen COMPLETE by
    logs/lbpoll_20260921-173830.log at 18:32 CEST = 16:32 UTC, still PENDING at 16:31 UTC
  - efficiency formula (research.md §1): score = AUC/(Base-maxAUC) + runtime_s/32400, minimised,
    Base ~0.5, maxAUC ~0.95  ->  -2.222*AUC + t/32400
Writes s6_budget.md."""
from __future__ import annotations

import json
from datetime import datetime

from common import OUT, PROJ

N = 1300
b = {k: json.loads((PROJ / f).read_text()) for k, f in {
    "eff": "qa/ensemble/time_eff_v1.json", "v2": "qa/stage8b/time_main_v2_today.json",
    "v3": "qa/stage8b/time_main_v3.json"}.items()}
s = {k: v["per_study_s_best"] for k, v in b.items()}
gpu = {k: min(v["runs"], key=lambda r: r["per_study_s"])["gpu_s_mean"] for k, v in b.items()}
a6000_min = {k: s[k] * N / 60 for k in s}

f_mvp = 25.0 / (0.53 * N / 60)
sub = datetime(2026, 9, 21, 14, 44, 43)
done_hi = datetime(2026, 9, 21, 16, 32)
v2_elapsed_max = (done_hi - sub).total_seconds() / 60
f_v2_max = v2_elapsed_max / a6000_min["v2"]
L = ["# S6 Kaggle runtime budget and efficiency score", "",
     f"A6000 bench, 1,300 studies: eff_v1 {a6000_min['eff']:.1f} min, main_v2 {a6000_min['v2']:.1f} min, "
     f"main_v3 {a6000_min['v3']:.1f} min (GPU s/study {gpu['eff']:.2f} / {gpu['v2']:.2f} / {gpu['v3']:.2f}).",
     f"Kaggle/A6000 factor: MVP (prep-bound) {f_mvp:.2f}; main_v2 (GPU-bound) <= {f_v2_max:.2f} "
     f"(submit->complete <= {v2_elapsed_max:.0f} min incl. queue).", ""]
lo, hi = a6000_min["v3"] * f_v2_max / 60, a6000_min["v3"] * f_mvp / 60
L.append(f"main_v3 projected hidden runtime: {lo:.1f} h (v2 factor, itself an upper bound) .. {hi:.1f} h "
         f"(MVP factor); NOT measured - read the real value off the Kaggle submission page.")
c_arm = (gpu["v3"] - gpu["v2"]) * N / 60  # A6000 GPU-min added by arm C (5 members, 5 views, z-TTA)
e_arm = gpu["eff"] * 1.5 * 2 * N / 60      # E inside main: K24 (x1.5 vs K16) and z-TTA (x2)
a_arm = (gpu["v2"]) * N / 60 - e_arm
L.append(f"Per-arm A6000 GPU-min (1,300 studies, z-TTA): A ~{a_arm:.0f}, E ~{e_arm:.0f}, C {c_arm:.0f}; "
         f"on Kaggle x{f_v2_max:.2f}: A ~{a_arm * f_v2_max / 60:.2f} h, C ~{c_arm * f_v2_max / 60:.2f} h.")
for rule, cap in (("9 h hard limit", 9.0), ("9 h with 25 % margin", 9.0 / 1.25), ("§2.7 rule: projected x1.25 < 6 h", 6.0 / 1.25)):
    room = cap - lo
    L.append(f"- {rule}: room after main_v3 = {room:.2f} h -> {max(room, 0) / (c_arm * f_v2_max / 60):.1f} more "
             f"C-sized arms or {max(room, 0) / (a_arm * f_v2_max / 60):.1f} A-sized arms (at the optimistic v3 estimate).")
L.append(f"- Dropping z-TTA on all 15 members halves GPU time: saves ~{gpu['v3'] / 2 * N / 60 * f_v2_max / 60:.2f} h; "
         "its AUC value has never been measured (needs one GPU OOF pass with/without the z-offset view).")
L.append("")

# Efficiency score
def eff_score(auc, minutes):
    return -auc / 0.45 + minutes * 60 / 32400

e_min = a6000_min["eff"] * f_mvp
L += ["## Efficiency track", "",
      "score = -AUC/0.45 + runtime_s/32400 (lower is better; research.md §1, Base 0.5, maxAUC 0.95 - both assumed).",
      f"1 min of runtime = {60 / 32400:.5f} = {60 / 32400 * 0.45:.5f} AUC; 0.01 AUC = {0.01 / 0.45 * 32400 / 60:.1f} min.", "",
      "| entry | LB AUC | runtime (est.) | score |", "|---|---|---|---|"]
entries = [("eff_v1 (E, 5 folds, K16)", 0.913, e_min), ("arm A alone", 0.931, (a_arm + 1.0 * N / 60 / 3) * f_v2_max),
           ("main_v2 (A+E)", 0.929, v2_elapsed_max), ("main_v3 (A+E+C)", 0.933, lo * 60)]
for n_, auc, m in entries:
    L.append(f"| {n_} | {auc:.3f} | {m:.0f} min | {eff_score(auc, m):.4f} |")
prep_min = b["eff"]["runs"][0]["prep_s_mean"] * N / 3 / 60
L += ["", f"eff_v1 is prep-bound: 3 workers x {b['eff']['runs'][0]['prep_s_mean']:.2f} s/study = {prep_min:.1f} A6000-min "
      f"of its {a6000_min['eff']:.1f}; GPU {gpu['eff'] * N / 60:.1f} min (overlapped). A GPU-free model would save at most "
      f"~{(a6000_min['eff'] - prep_min) * f_mvp:.1f} Kaggle-min = {(a6000_min['eff'] - prep_min) * f_mvp * 60 / 32400 * 0.45:.4f} AUC-equivalent.",
      "Break-even AUC for a replacement at runtime t (same score as eff_v1 at LB 0.913):"]
for t in (5, 10, 15, e_min, 30, 45):
    need = 0.45 * (0.913 / 0.45 - (e_min - t) * 60 / 32400)
    L.append(f"- t = {t:.0f} min -> AUC >= {need:.4f} ({need - 0.913:+.4f})")
(OUT / "s6_budget.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))

# Estimate for a C-type efficiency entry: 2 folds, K16, no z-TTA, 4 decoded series instead of E's 3.
c_gpu_min = c_arm * (2 / 5) / 2 / 1.5
c_prep_min = prep_min * 4 / 3
c_a6000 = max(c_gpu_min, c_prep_min) + 0.25 * min(c_gpu_min, c_prep_min)
c_kaggle_lo, c_kaggle_hi = c_a6000 * f_v2_max, c_a6000 * f_mvp
need = [0.913 - 0.45 * (e_min - t) * 60 / 32400 for t in (c_kaggle_lo, c_kaggle_hi)]
extra = ["", f"C-type efficiency candidate (2 folds, K16, no TTA; est.): GPU {c_gpu_min:.1f} + prep {c_prep_min:.1f} A6000-min "
         f"-> ~{c_a6000:.1f} A6000-min -> {c_kaggle_lo:.0f}-{c_kaggle_hi:.0f} Kaggle-min; break-even LB AUC "
         f"{need[0]:.4f}-{need[1]:.4f}."]
with open(OUT / "s6_budget.md", "a", encoding="utf-8") as fh:
    fh.write("\n".join(extra) + "\n")
print("\n".join(extra))
