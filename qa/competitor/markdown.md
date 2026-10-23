===== md cell 0 =====
# RSNA Knee | DINOsaur V5? 🦖

### 🙏 Thanks & provenance

- **Mattia Angeli** — `mattiaangeli/bend-the-knee-to-speedy-raptors`
- **André Luiz Pedroso** — `andreluizpedroso/rsna-knee-tony-rank-ensemble`
- **Dread Development** — `dreaddevelopment/raptor-knee-finespacing`
- plus the upstream DINOv2 / DINOv3 / RadImageNet / CoAt contributors already
  credited by the parent notebook.

This is a derived competition experiment, not a relabeling of the upstream work.
The original model weights, code and datasets keep their original authorship and
licenses.

===== md cell 1 =====
## 0. Run configuration

===== md cell 3 =====
## Exploratory analysis

Read-only, a few minutes, no effect on the submission. Skip to the pipeline if that is all you came for.

===== md cell 11 =====
## Environment, assets and the speed harness

Thread limits, asset discovery by content hash, and the child-process runners for the two CoAt family members. Nothing here touches a prediction.

===== md cell 13 =====
## Dense depth allocation

How the D4 member splits each anatomical slot into depth zones, and how the per-study slice budget is divided between them.

===== md cell 15 =====
## Pipeline startup

Paths, targets, device inventory and the phase log.

===== md cell 18 =====
## Reading DICOM headers

Series-level metadata without touching pixels: sequence description, contrast weighting, fat suppression and laterality.

===== md cell 20 =====
## Slot selection

One series per slot per study — plane x contrast x fat suppression. A fixed tensor out of a variable bag of series.

===== md cell 22 =====
## Geometric ordering

Files sorted by physical slice position rather than name, because the weights were fitted against that order.

===== md cell 25 =====
## Building the cache

Decode once, hold as uint8, normalise laterality so every knee presents as left.

===== md cell 27 =====
## Stage 1 — per-finding attention over slots

Each finding gets its own attention weights over the six slots, seeded with the slots a radiologist would actually look at.

===== md cell 31 =====
## Stage 1 — the twenty-member ensemble

Shared prefix loading, fingerprint gating, and the rank-mean combine. All twenty members must match or the run aborts.

===== md cell 35 =====
## Stage 2 — A5 attention-pooling folds

===== md cell 38 =====
## Stage 2 — pooling architectures

===== md cell 42 =====
## Stage 3 — RadImageNet with the 88-feature calibrator

===== md cell 44 =====
## Stage 4 — Raptor views and the CoAt family

Four views at 94 windows, then two CoAt children (residual resgated top-3, D4 depth-zone SWA3) combined by equal rank averaging and mixed in at 0.40. Both children fail soft.

===== md cell 46 =====
## Diagnostics — is the routing doing anything?

===== md cell 48 =====
## Publish

`submission.csv` appears once, here, after the whole graph has succeeded.

===== md cell 50 =====
## DINOsaur V5 — Fine-Spacing residual injection

The parent pipeline has already written its exact 0.942 `submission.csv` before
this cell runs.

Fine-Spacing v9 is deliberately treated as a **residual source**, not as another
global Raptor weight. The public Raptor prediction already inside the 0.942 graph
is subtracted in rank space, leaving only the ordering change contributed by the
80-slice / 78-window view.

Outputs:

- `submission_dinosaur_v5_0942_anchor.csv` — exact parent control
- `submission_dinosaur_v5_finespacing_safe.csv` — residual scale 0.06
- `submission_dinosaur_v5_finespacing_main.csv` — residual scale 0.10 (**default**)
- `submission_dinosaur_v5_finespacing_strong.csv` — residual scale 0.16
- `dinosaur_v5_finespacing_diagnostics.json`

No existing DINO, A5, RadImageNet, Raptor, residual-CoAt or D4 arithmetic is
changed before this point.


===== md cell 52 =====
---

## Credits

[Jiwei Liu](https://www.kaggle.com/code/jiweiliu/rsna-knee-fast-2xt4-inference) · 
[Pilkwang](https://www.kaggle.com/datasets/pilkwang/rsna-knee-weights) · Sofia Anjenje · Antoine G. · prvsiyan · Marwan Mahmoud · dreaddevelopment / Roman Tamrazov · [renta.k](https://www.kaggle.com/code/renta0426/rsna-knee-hybrid-raptor-lateral-meniscus-r100) · [Anvith Pothula](https://www.kaggle.com/code/anvithpothula/rsna-base)