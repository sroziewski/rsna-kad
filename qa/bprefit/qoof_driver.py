"""bprefit step 1 (2026-10-21): out-of-fold predictions for Qile's g1 MTR / MTH / OLPS / OLPC with bpblur's own test-time
path (B's run_ours + patch_qile_src 0.65 px after the warp on every view, HIRES_BLUR_PX 0), so stacker5 can be refitted
on what bpblur's stacker actually sees. Modelled on andreytikhomirov/rsna-knee-mth-blur-oof; runs inside a fake /kaggle
tree (the ours scripts hardcode /kaggle/input), built by run_qoof.sh.

  parity  MTH, fold 0, 40 non-gold fold-0 studies: the unpatched script must reproduce the mthoof sharp parity file;
          the patched one must differ from it (blur live)
  oof     fold k model on fold k's non-gold studies (folds_v1), k = 0..4
  gold    all 5 folds on the 58 gold studies (is_gold), as the kernel scores the test set
"""
import argparse
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

MODELS = "MTR,MTH,OLPS,OLPC"
ap = argparse.ArgumentParser()
ap.add_argument("--work", required=True)
ap.add_argument("--comp", required=True)
ap.add_argument("--folds", required=True)
ap.add_argument("--wheel", required=True)
ap.add_argument("--ours-sharp", required=True)
ap.add_argument("--ours-blur", required=True)
ap.add_argument("--parity-ref", required=True)
a = ap.parse_args()
W = a.work
pkgs = f"{W}/pkgs"
if not os.path.isdir(pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--no-index", "-q", "--no-cache-dir",
                    "--disable-pip-version-check", "--target", pkgs, a.wheel], check=True)
F = pd.read_csv(a.folds, dtype={"StudyInstanceUID": str})
gold = F[F.is_gold].StudyInstanceUID.tolist()
F = F[~F.is_gold]
assert len(F) == 4349 and len(gold) == 58, (len(F), len(gold))


def run(tag, script, models, folds, ids):
    lst = f"{W}/ids_{tag}.csv"
    pd.DataFrame({"StudyInstanceUID": sorted(ids)}).to_csv(lst, index=False)
    t = time.time()
    env = dict(os.environ, KNEE_MODELS=models, KNEE_FOLDS=folds, KNEE_COMP=a.comp, KNEE_SPLIT="train", KNEE_STUDY_LIST=lst,
               KNEE_HIRES_BLUR_PX="0", KNEE_OUT=f"{W}/{tag}.csv", KNEE_GPU_THREADS="1", KNEE_BACKBONE_CHUNK="48",
               PYTHONPATH=pkgs + os.pathsep + os.environ.get("PYTHONPATH", ""))
    with open(f"{W}/log_{tag}.txt", "w") as log:
        rc = subprocess.run([sys.executable, script], env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    print("DONE", tag, "models", models, "folds", folds, "studies", len(ids), "rc", rc, f"{time.time() - t:.0f}s", flush=True)
    if rc:
        sys.exit(f"{tag} failed, see log_{tag}.txt")


def mth(tag):
    return pd.read_csv(f"{W}/{tag}_MTH_foldmean.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID").sort_index()


f0 = sorted(F[F.fold == 0].StudyInstanceUID)[:40]
ref = pd.read_csv(a.parity_ref, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID").sort_index()
run("par_sharp", a.ours_sharp, "MTH", "0", f0)
run("par_b065", a.ours_blur, "MTH", "0", f0)
d_sharp = np.abs(mth("par_sharp").values - ref.loc[mth("par_sharp").index].values)
d_blur = np.abs(mth("par_b065").values - ref.loc[mth("par_b065").index].values)
print(f"parity sharp vs mthoof ref: max {d_sharp.max():.5f} mean {d_sharp.mean():.5f}; 0.65 vs ref: max {d_blur.max():.4f} "
      f"mean {d_blur.mean():.4f}", flush=True)
if d_sharp.max() > 0.01 or d_blur.mean() < 0.003:
    sys.exit("parity failed: the setup does not reproduce the sharp reference, or the blur is not live")
for k in range(5):
    run(f"oof_f{k}", a.ours_blur, MODELS, str(k), F[F.fold == k].StudyInstanceUID.tolist())
run("gold", a.ours_blur, MODELS, "0,1,2,3,4", gold)
print("ALL_DONE", flush=True)
