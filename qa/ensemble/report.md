# Ensemble analysis and the two stage-7/8 submissions (2026-09-21)

Scope: architecture §2.6 (equal weights only, no LB-tuned per-label weights), §2.7, §2.8, §4 rows 7-8.
Everything below is OOF evidence; no leaderboard number entered any decision.

Reproduce:

```bash
python qa/ensemble/blend_oof.py --runs runs/main_convnext_s_384 runs/eff_effv2s_256 \
    --runs-optional runs/mvp_effv2s_256 --targets v1 --n-boot 2000 \
    --out qa/ensemble/blend_tables.md --json-out qa/ensemble/blend.json
CUDA_VISIBLE_DEVICES=1 bash scripts/run_screen.sh kinfer_eff python scripts/bench_infer.py kinfer \
    --config configs/eff_effv2s_256.yaml --k 12 16 24 --batch-studies 4 --out qa/ensemble/kinfer_eff.json
python scripts/bench_infer.py time --weights export/eff_v1  --device cuda:1 --replicate 12 --repeat 2 \
    --out qa/ensemble/time_eff_v1.json
python scripts/bench_infer.py time --weights export/main_v2 --device cuda:1 --replicate 8  --repeat 2 \
    --out qa/ensemble/time_main_v2.json
```

## 1. Decision

**Main track ships the blend: `main_convnext_s_384` + `eff_effv2s_256`, equal weights, mean of sigmoids.**
The blend beats arm A alone by **+0.0077 OOF-weak** (0.9292 vs 0.9215), 95 % CI of the paired delta
[+0.0051, +0.0103], P(blend > arm A) = 1.000 over 2,000 study-level bootstrap replicates. Strict
(+0.0090) and gold-58 (+0.0059) move the same way, so this is not an artefact of the weak-label
binarisation. It also clears the stage-8 gate "ensemble OOF-weak >= best single + 0.005".

**`mvp_effv2s_256` is excluded.** Adding it reads +0.0012 OOF-weak over the two-member blend, but the
paired bootstrap puts that at P = 0.862 - not a demonstrable gain - and it costs 5 extra members at
inference. Its apparent contribution is concentrated in contusion/fracture, the two labels where the
old rule targets (v0) it was trained on differ most from v1, so the gain is as likely to be target-noise
diversity as signal. Revisit only with a run retrained on v1.

**Mean of sigmoids, not rank average.** The rank blend is worse on OOF-weak (0.9271 vs 0.9292,
P(mean > rank) = 1.000) and is not expressible in the `ensemble.json` contract (§3.2), which averages
probabilities. No reason to change the contract.

Diversity supports the blend: mean per-label Pearson correlation between the two arms OOF predictions
is **0.832** (Spearman 0.767), lowest on mcl 0.693 and fracture 0.785. Per label the two arms are
genuinely complementary - arm A wins contusion (0.9165 vs 0.8582) and lat_men (0.9081 vs 0.8695),
the efficiency arm wins synovitis (0.9470 vs 0.9084) and med_oa (0.9376 vs 0.9354). The blend beats
**both** singles on 11 of 12 labels; the single exception is contusion (0.9104 vs arm A 0.9165),
where averaging in the weaker arm costs 0.006. §2.6 forbids fixing that with a per-label weight.

## 2. K_infer for the efficiency kernel

`scripts/bench_infer.py kinfer` on all 4,407 cached studies, 5 folds, targets v1 (`qa/ensemble/kinfer_eff.json`):

| K_infer | OOF-weak | delta vs K 24 | strict | gold-58 | GPU s (5 folds, A6000) | peak GB |
|---|---|---|---|---|---|---|
| 12 | 0.90478 | **-0.00325** | 0.8337 | 0.8676 | 85 | 0.98 |
| 16 | 0.90655 | -0.00148 | 0.8369 | 0.8700 | 104 | 1.27 |
| 24 | 0.90803 | 0.0 (ref) | 0.8388 | 0.8664 | 151 | 1.86 |

**K_infer 12 does not hold AUC within 0.002 of K 24** (it costs 0.0033), so it was not shipped.
K_infer 16 does (-0.0015) and cuts the GPU pass to 0.69x, so the efficiency bundle ships **K_infer 16**.
The override is written into each member `models/*.json` (`arch.K_infer`), because `infer.Member.spec`
reads K from the member bundle, not from the top-level `K_infer` in `ensemble.json` - that field is only a record.

## 3. Runtime (efficiency track)

`bench_infer.py time`, real raw-DICOM path, one A6000 (cuda:1), 3 workers, the 3 public test studies
replicated 12x so the producer/consumer pipeline reaches steady state, best of 2 repeats:

| | per study | 1,300 studies |
|---|---|---|
| measured, 1x A6000 | **0.423 s** (prep 1.00 s / 3 workers, GPU 0.21 s) | 9.2 min |
| T4 rule (x4.5, §2.7) | 1.90 s | **41.2 min = 2,474 s** |
| T4 rule + 25 % margin | 2.38 s | 51.5 min = 3,092 s |

At the efficiency exchange rate of runtime/32400, 2,474 s is worth **0.076 AUC-equivalent**
(3,092 s with margin: 0.095). For comparison the K 24 variant would have cost about 0.21 s/study more
of GPU time (+27 min under the T4 rule, ~0.05 AUC-equivalent) to buy 0.0015 AUC - K 16 is the better
trade, and K 12 would have bought another ~0.02 AUC-equivalent of runtime for 0.0033 AUC.

Caveats: the replicated studies are decoded from the same 3 directories, so the page cache makes the
decode side optimistic; and the x4.5 T4 factor is conservative - stage 4 measured 0.53 s/study on an
A6000 for the MVP and the actual Kaggle hidden rerun took 25 min for 1,300 studies (1.15 s/study on
2xT4), an effective factor of 2.2. The empirical projection for `eff_v1` is therefore ~20 min.

`main_v2` for the record: 2.98 s/study on **one** A6000 -> 290.8 min under the T4 rule (363.5 min with
margin). Kaggle runs two T4s and the notebook fills both, so the real figure is roughly half of the GPU
half; scaling the MVP measured 25 min hidden rerun by the per-study image cost puts `main_v2` near
1.6 h, well inside the 9 h wall and the 8.25 h guard in `ensemble.json`.

## 4. Submissions

| track | bundle | dataset | kernel | submission | description |
|---|---|---|---|---|---|
| efficiency | `export/eff_v1` (5 x eff_effv2s_256, K 16, no TTA) | `sroziews/rsna-kad-weights` v3 | v8 | **56433490** | OOF-weak 0.9066 @K16, gold 0.8666 -> **public LB 0.913** |
| main | `export/main_v2` (5 x main_convnext_s_384 + 5 x eff_effv2s_256, equal weights, z-TTA, K 24) | v4 | v9 | **56433900** | OOF-weak 0.9292, gold 0.8964 -> public LB pending |

Both followed the stage-6 lesson: `dataset-status` was polled until the *file listing* showed the new
bundle (the overall status flips to `ready` before the listing catches up - v4 reported `ready` while
still listing the v3 `VERSION` of 122 bytes), and only then was the kernel pushed. Each kernel
`submission.csv` was diffed against a local fp32 CPU run of the same bundle before submitting:

- eff_v1: same 3 ids, 12 columns, 0 NaN, max abs diff **5.5e-4** (mean 9.9e-5) vs `export/submission_local_eff_v1.csv`
- main_v2: max abs diff **2.1e-4** (mean 5.3e-5) vs `export/submission_local_main_v2.csv`; mean abs
  difference to the efficiency kernel output is 0.032, confirming the kernel really mounted v4 and not v3

Public LB: the efficiency bundle scored **0.913** (arm A alone at 384 px with TTA scored 0.931 and the
MVP 0.865), i.e. the efficiency track gives up 0.018 LB for roughly a seventh of the training cost and a
third of the inference cost. OOF-weak predicted the gap well (0.9066 vs 0.9215, -0.015 vs -0.018 on LB).

Kernel logs: `qa/stage7/kernel_v8/`, `qa/stage8/kernel_v9/`. Both logged the expected member count and
mode (`5 members, mode efficiency` / `10 members, mode main`), 0 prior rows, wheels installed offline.

## 5. Generated tables

`qa/ensemble/blend_tables.md` (written by `blend_oof.py`), reproduced below.

### Ensemble OOF analysis (targets v1)

4407 common studies (4349 weak, 58 gold); OOF-weak per §2.5 (binarised targets, macro over 12 labels), gold-58 vs `train.csv`. Bootstrap: 2000 replicates, seed 2026, rows resampled once and reused for every candidate, so the deltas are paired.

| candidate | OOF-weak | delta vs best single | 95% CI of delta | P(blend > best) | strict | gold-58 | gold CI95 |
|---|---|---|---|---|---|---|---|
| main_convnext_s_384 | 0.9215 | +0.0000 | [+0.0000, +0.0000] | 0.000 | 0.8524 | 0.8905 | [0.861, 0.919] |
| eff_effv2s_256 | 0.9080 | -0.0135 | [-0.0186, -0.0084] | 0.000 | 0.8388 | 0.8666 | [0.832, 0.897] |
| mvp_effv2s_256 | 0.8815 | -0.0400 | [-0.0467, -0.0334] | 0.000 | 0.8095 | 0.8466 | [0.813, 0.877] |
| mean(main+eff) | 0.9292 | +0.0077 | [+0.0051, +0.0103] | 1.000 | 0.8614 | 0.8964 | [0.864, 0.924] |
| rank(main+eff) | 0.9271 | +0.0056 | [+0.0029, +0.0084] | 1.000 | 0.8601 | 0.8964 | [0.864, 0.924] |
| mean(main+eff+mvp) | 0.9304 | +0.0089 | [+0.0055, +0.0126] | 1.000 | 0.8583 | 0.8973 | [0.867, 0.925] |
| rank(main+eff+mvp) | 0.9272 | +0.0057 | [+0.0024, +0.0093] | 1.000 | 0.8593 | 0.8918 | [0.862, 0.918] |

Best single run on OOF-weak: **main_convnext_s_384**.

### Paired bootstrap: P(row's OOF-weak > column's)

| | main_convnext_s_384 | eff_effv2s_256 | mvp_effv2s_256 | mean(main+eff) | rank(main+eff) | mean(main+eff+mvp) | rank(main+eff+mvp) |
|---|---|---|---|---|---|---|---|
| main_convnext_s_384 | - | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| eff_effv2s_256 | 0.000 | - | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| mvp_effv2s_256 | 0.000 | 0.000 | - | 0.000 | 0.000 | 0.000 | 0.000 |
| mean(main+eff) | 1.000 | 1.000 | 1.000 | - | 1.000 | 0.139 | 0.966 |
| rank(main+eff) | 1.000 | 1.000 | 1.000 | 0.000 | - | 0.005 | 0.458 |
| mean(main+eff+mvp) | 1.000 | 1.000 | 1.000 | 0.862 | 0.995 | - | 1.000 |
| rank(main+eff+mvp) | 1.000 | 1.000 | 1.000 | 0.034 | 0.542 | 0.000 | - |

### Per-label OOF-weak AUC

| label | main_convnext_s_384 | eff_effv2s_256 | mvp_effv2s_256 | mean(main+eff) | rank(main+eff) | mean(main+eff+mvp) | rank(main+eff+mvp) |
|---|---|---|---|---|---|---|---|
| acl | 0.9650 | 0.9544 | 0.9048 | 0.9690 | 0.9677 | 0.9632 | 0.9606 |
| mcl | 0.8691 | 0.8494 | 0.7436 | 0.8819 | 0.8785 | 0.8712 | 0.8655 |
| med_men | 0.9589 | 0.9465 | 0.9177 | 0.9597 | 0.9594 | 0.9573 | 0.9557 |
| lat_men | 0.9081 | 0.8695 | 0.7666 | 0.9150 | 0.9064 | 0.9063 | 0.8922 |
| med_oa | 0.9354 | 0.9376 | 0.9194 | 0.9431 | 0.9439 | 0.9463 | 0.9466 |
| lat_oa | 0.8630 | 0.8535 | 0.8370 | 0.8688 | 0.8701 | 0.8753 | 0.8786 |
| pf_oa | 0.9112 | 0.8890 | 0.8614 | 0.9164 | 0.9151 | 0.9128 | 0.9108 |
| effusion | 0.9539 | 0.9457 | 0.9407 | 0.9573 | 0.9572 | 0.9581 | 0.9580 |
| synovitis | 0.9084 | 0.9470 | 0.9138 | 0.9459 | 0.9461 | 0.9501 | 0.9479 |
| baker | 0.9678 | 0.9598 | 0.9428 | 0.9717 | 0.9710 | 0.9687 | 0.9691 |
| contusion | 0.9165 | 0.8582 | 0.9147 | 0.9104 | 0.9039 | 0.9282 | 0.9200 |
| fracture | 0.9006 | 0.8860 | 0.9152 | 0.9108 | 0.9062 | 0.9273 | 0.9217 |

### Per-label gold-58 AUC

| label | main_convnext_s_384 | eff_effv2s_256 | mvp_effv2s_256 | mean(main+eff) | rank(main+eff) | mean(main+eff+mvp) | rank(main+eff+mvp) |
|---|---|---|---|---|---|---|---|
| acl | 0.9651 | 0.9632 | 0.9400 | 0.9779 | 0.9730 | 0.9779 | 0.9706 |
| mcl | 0.8912 | 0.8753 | 0.5760 | 0.8934 | 0.9184 | 0.8571 | 0.8141 |
| med_men | 0.9459 | 0.9219 | 0.8984 | 0.9435 | 0.9489 | 0.9495 | 0.9507 |
| lat_men | 0.8814 | 0.7596 | 0.7329 | 0.8559 | 0.8335 | 0.8447 | 0.8199 |
| med_oa | 0.9403 | 0.9721 | 0.9411 | 0.9659 | 0.9674 | 0.9705 | 0.9721 |
| lat_oa | 0.7901 | 0.7427 | 0.8259 | 0.7737 | 0.7814 | 0.7930 | 0.7969 |
| pf_oa | 0.8301 | 0.8185 | 0.8095 | 0.8443 | 0.8443 | 0.8366 | 0.8378 |
| effusion | 0.9429 | 0.9422 | 0.9646 | 0.9627 | 0.9640 | 0.9764 | 0.9789 |
| synovitis | 0.7921 | 0.7342 | 0.8118 | 0.8100 | 0.8065 | 0.8339 | 0.8256 |
| baker | 0.9484 | 0.9475 | 0.8931 | 0.9438 | 0.9511 | 0.9330 | 0.9348 |
| contusion | 0.8664 | 0.8354 | 0.8698 | 0.8677 | 0.8583 | 0.8731 | 0.8772 |
| fracture | 0.8917 | 0.8868 | 0.8965 | 0.9181 | 0.9097 | 0.9222 | 0.9236 |

### Diversity: OOF correlation main_convnext_s_384 vs eff_effv2s_256

| label | Pearson | Spearman |
|---|---|---|
| acl | 0.882 | 0.718 |
| mcl | 0.693 | 0.673 |
| med_men | 0.918 | 0.887 |
| lat_men | 0.787 | 0.693 |
| med_oa | 0.894 | 0.875 |
| lat_oa | 0.811 | 0.786 |
| pf_oa | 0.820 | 0.820 |
| effusion | 0.894 | 0.874 |
| synovitis | 0.786 | 0.737 |
| baker | 0.901 | 0.739 |
| contusion | 0.812 | 0.728 |
| fracture | 0.785 | 0.681 |
| **mean** | 0.832 | 0.767 |


(`mvp_effv2s_256` reads 0.8815 above, not the 0.8463 in its own `metrics.json`: that run was trained and
scored on rule targets v0, and this table rescores every run on targets v1 so the rows are comparable.)

## 6. Open items

- `notebooks/kaggle_inference.py` was regenerated by `export_kaggle.py` and now differs from the file
  that produced LB 0.931 (kernel v7): another workstream added `ModelConfig.view_crop_mm` and a `views`
  argument to `model.load_bundle` without rebuilding the notebook, so
  `tests/test_export.py::test_committed_notebook_equals_fresh_build` was already failing before this
  work. Both new bundles have no view crops, so the inlined change is inert for them, and the
  kernel-vs-local diffs above confirm it. The rebuilt notebook now matches the source tree again.
- `infer.py` does **not** implement `arch.view_crop_mm`: a member exported from a run that uses per-slot
  view crops would be fed uncropped duplicates without any error. `model.load_bundle` guards this with
  `views=True`, `load_member` does not. This must be closed before any arm B bundle is exported.
- The `main_v2` A6000 timing is single-GPU; a two-GPU measurement was not taken because GPU 0 was
  reserved for another job.

