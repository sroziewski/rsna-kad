#!/usr/bin/env python3
"""CPU watcher for the triage chain: once per_label_attn fold 0 has finished epoch 6 (one epoch
before its end, i.e. before the chain starts the no-prior run), compare it with
per_label_residual at the SAME epoch on the run's own weak AUC and point
configs/main_armn_noprior_384.yaml at the better head. Picks an ablation head only; the
decision on the head itself is made by qa/stage11/compare.sh on the v1 yardstick."""
import json, re, sys, time
from pathlib import Path

EP = 6
SFX = sys.argv[1] if len(sys.argv) > 1 else ""
def hist(run):
    p = Path(f"runs/{run}/fold0/metrics.json")
    try:
        return json.loads(p.read_text())["history"]
    except Exception:
        return []

while len(hist(f"main_armn_perlabel{SFX}_384")) <= EP:
    time.sleep(60)
r, p = hist(f"main_armn_residual{SFX}_384")[EP]["oof_weak_auc"], hist(f"main_armn_perlabel{SFX}_384")[EP]["oof_weak_auc"]
head = "per_label_attn" if p > r else "per_label_residual"
cfg = Path(f"configs/main_armn_noprior{SFX}_384.yaml")
t = cfg.read_text()
t = re.sub(r"^  head: \S+$", f"  head: {head}", t, flags=re.M)
t = re.sub(r"^# Head set to .*$", f"# Head set to {head} after the fold-0 triage (epoch-{EP} weak: residual {r:.4f}, perlabel {p:.4f}).", t, flags=re.M)
cfg.write_text(t)
print(f"epoch {EP}: residual {r:.4f} perlabel {p:.4f} -> noprior head = {head}")
