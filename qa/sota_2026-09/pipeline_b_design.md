# Pipeline B - design (architect, 2026-09-30)

Purpose: a model family built to be **merged** with the A entry (main_v5_C = 6 Cfull ConvNeXt-V2-tiny, public LB
0.948), making different errors from it. Inputs read in full: `qa/sota_2026-09/research.md` (starting point, not a
mandate), `docs/architecture.md` §2-§4.5, and `src/rsna_kad/{config,data,model,train,infer,cache,metrics}.py`,
`labels/{targets,pseudo}.py`, `scripts/{train,export_kaggle}.py`. Estimates are marked *est.*; everything marked
[V] was checked against the primary source today.

**One-line design.** B = DINOv3 ViT-B/16 (SSL, LVD-1689M) at 288 px over 5 views (cor at 110 mm), with the existing
per-label attention head (no prior) and 12 + 24 auxiliary report findings from a new aux-only LLM pass. B has its own
chain: round 0 on report targets, then B's own 2-seed OOF teacher, then 4 full-data students. It uses nothing A
produced. Merge = 0.5 x mean(A) + 0.5 x mean(B), fixed now.

---

## 0. Independence ledger (what differs, what is deliberately shared)

| Axis | A (main_v5_C) | B | Independence |
|---|---|---|---|
| Pretraining objective / data | FCMAE masked autoencoding, IN-22k fine-tuned | DINOv3 self-distillation from ViT-7B, LVD-1689M web images, no ImageNet labels | high |
| Family | ConvNeXt (conv, 7x7 depthwise) | ViT-B/16, RoPE, 4 register tokens | high |
| Teacher (half of every target) | main_v3 OOF blend (A's arms) | **mean OOF of B's own round-0 models (2 seeds x 5 folds)** | high - this was the measured collapse source |
| Target space | 12 keys | 12 keys + 24 auxiliary findings (new extraction pass) | medium-high |
| Head | `shared_attn` (gated MIL per slot + concat MLP) | `per_label_attn`, `slot_prior_strength: 0` (12+24 attention distributions over the whole slot x slice bag) | medium |
| Input | 384 px, K 12 train / 24 infer, views 130/130/130/130 + sag 90 mm | 288 px, K 12 train / 16 infer, views sag 130, **cor 110**, ax 130, sag_t1 130, sag 90 | low-medium |
| Round-0 loss on silent cells | weight 0.1, calibrated target | same + one-sided large-loss rejection (§4.3) | low |
| 12-key label table | `v1_1_gemma3` (Qwen V1+V2 + rules + Gemma-3) | **same table, shared on purpose** (§4.1) | none |

**Shared, for honest evaluation and one decode**: `data/folds/folds_v1.csv` (md5 `4dc54b74...`); gold-58 excluded
from every B training table *and* from B's teacher; `scripts/evaluate_runs.py`, the paired-bootstrap helpers in
`metrics.py`, and the §4.2/§4.5 rules; cache v1 (384 px, 130 mm, uint8, per-series p1/p99). B derives its 288 px
inputs from the 384 px cache (and from the same `StudyPreprocessor` volumes in the kernel) through the existing
`read_triplets -> warp_slices` INTER_AREA path. **Design decision: no second cache and no second decode.** The kernel
decodes each study once, and `predict_studies` already builds one input tensor per distinct `InputSpec`.

---

## 1. Backbone, weight id, licence

**Choice: `vit_base_patch16_dinov3.lvd1689m` (timm; HF repo `timm/vit_base_patch16_dinov3.lvd1689m`).**
Verified [V] from HF API and the repo files today:
- The repo is **not gated** (`gated: False`), unlike `facebook/dinov3-*`, which is manual-gated.
- The repo ships `LICENSE.md` with md5 `712846e16e4d72d12ae1052fe7e00133`. That is byte-identical to
  `github.com/facebookresearch/dinov3/blob/main/LICENSE.md` ("DINOv3 License", last updated August 19, 2025).
- `config.json`: 85.6 M params, **23.6 GMACs at 256 px**, `fixed_input_size: false`, ImageNet mean/std (the same
  constants as `model.normalize`), `global_pool: avg`.
- timm source (`timm/models/eva.py`): `dynamic_img_size=True`, `num_reg_tokens=4`, `qkv_bias=False`, RoPE. `avg` pooling
  excludes the 5 prefix tokens (`global_pool_nlc(..., num_prefix_tokens)`), `set_grad_checkpointing` exists, and attention
  uses SDPA. **288 px therefore needs no model code: `KneeMIL`'s `timm.create_model(cfg.backbone, num_classes=0,
  global_pool="avg", drop_path_rate=...)` works as is.**
- The model first appears in **timm 1.0.20 (2025-09-21, PyPI)**. The server venv and the Kaggle image must both be
  >= 1.0.20 (§8 item 7).

**Licence: DINOv3 License. Verbatim clauses (primary source, fetched 2026-09-30):**
- Grant, 1.a: *"You are granted a non-exclusive, worldwide, non-transferable and royalty-free limited license under
  Meta's intellectual property or other rights owned by Meta embodied in the DINO Materials to use, reproduce,
  distribute, copy, create derivative works of, and make modifications to the DINO Materials."*
- Derivatives and redistribution, 1.b.i: *"Distribution of DINO Materials, and any derivative works thereof, are
  subject to the terms of this Agreement. If you distribute or make the DINO Materials, or any derivative works
  thereof, available to a third party, you may only do so under the terms of this Agreement and you shall provide a
  copy of this Agreement with any such DINO Materials."*
- Ownership of derivatives, 5.a: *"with respect to any derivative works and modifications of the DINO Materials that are
  made by you, as between you and Meta, you are and will be the owner of such derivative works and modifications."*
- Publication, 1.b.ii: *"If you submit for publication the results of research you perform on, using, or otherwise in
  connection with DINO Materials, you must acknowledge the use of DINO Materials in your publication."*
- Use limits, 1.b.v: no ITAR / *"military or warfare purposes, nuclear industries or applications, espionage, or the
  development or use of guns or illegal weapons"*. **No commercial-use restriction and no no-derivatives clause.**
- 8: Meta may amend the terms, and continued use means acceptance.

**Verdict: USABLE.** The licence allows derivative works and allows redistributing them, commercially included, so the
no-derivatives problem does not arise. It is not CC-BY-NC: fine-tuned B weights must be redistributed under the DINOv3
Agreement with a copy attached, and cannot be relicensed as CC-BY-NC. That is the case rule 5.a.4 covers. As recorded
in architecture §4.3 from the team's direct read of the rules (2026-09-25): *"In the event that input data or pretrained
models with an incompatible license are used to generate your winning solution, you do not need to grant an open
source license ... for that data and/or model(s)."* Winner licence: *"6. WINNER LICENSE TYPE CC-BY-NC 4.0"*.
**Open item (GL, 10-01, 15 min):** I could not re-read the rules page today (it renders client-side, so the fetch
returned only the title). Under our standing rule the developer re-reads rule 5.a.4 and the external-models clause
(publicly available, minimal cost; the timm repo is free and ungated) from the Kaggle rules text before any B
training starts. What ships: our code stays CC-BY-NC; the B weights go in the dataset together with
`LICENSES/DINOv3_LICENSE.md` and a `NOTICE` naming the DINOv3-derived members, and the write-up acknowledges DINOv3
(1.b.ii).

**Fallbacks (pre-declared order):**
1. If G0 fails (instability or throughput): `convnext_small.dinov3_lvd1689m` (timm, same DINOv3 licence, `gated: False`
   [V]) at 384 px. It keeps the pretraining independence but is the same family as A.
2. If GL fails (the rules text turns out to reject non-CC-BY-NC derivatives): `vit_base_patch14_reg4_dinov2.lvd142m`
   (timm, **Apache-2.0** per card tag [V], `gated: False` [V]; DINOv2 repo LICENSE is Apache [V]) at **280 px**
   (patch 14 needs 20x20 = 400 tokens, *est.* 1.3x ViT-B/16@288 FLOPs).

MedSigLIP (400 M, *est.* ~8x C per image) and RadImageNet (224 px CNN) are not needed and do not fit the calendar.

Inference never touches HF. The kernel builds `KneeMIL` with `pretrained=False` from our fine-tuned safetensors, which
already contain the backbone, so **the offline requirement is met by the weights dataset alone.** Only the server
downloads the pretrained weights, once, into `HF_HOME=/raid/sroziewski/cache/huggingface` via `run_screen.sh` (no home-dir
writes).

---

## 2. Input

- **Views** (`slots` / `view_crop_mm`): `[sag_fs, cor_fs, ax_fs, sag_t1, sag_fs]` / `[130, 110, 130, 130, 90]`.
  - The coronal view is 110 mm, not 130 (A) and not 100 (arm D). §4.1 measured that a centred crop keeps the MCL in
    100 % of studies at 110 mm but only 44 % at 100 mm. 110 mm gives a 1.18x zoom where A's worst labels
    (lat_oa, mcl) are read.
  - The 90 mm sagittal view at 288 px is 0.31 mm/px, which is native resolution. 130 mm at 288 px is 0.45 mm/px,
    coarser than A's 0.34.
  - Rejected: a 90 mm axial PF crop (unmeasured, and the patella sits off-centre); `cor_t1` (rejected in §4.1, missing
    in 22.7 %).
- **Resolution**: 288 px (18x18 = 324 patch tokens + 5 prefix = 329 tokens). 256 px would be 1.27x cheaper but coarser
  than A everywhere; 288 fits the budget (§2.1).
- **Slices**: triplets `[i-1, i, i+1]` as RGB, the existing path. Train K = 12 stratified-random (like C). Infer
  **K = 16 uniform strata**: for the typical 30-slice series the centres are about 2 slices apart, so the triplets
  cover every slice. That is the "full coverage" of research R4 without a ragged-tensor data path.
- **Normalisation**: the cache's per-series p1/p99 uint8, then /255 and ImageNet mean/std on the GPU
  (`model.normalize`). DINOv3's own `pretrained_cfg` uses the same constants [V].

### 2.1 T4 runtime budget (per study, from FLOPs; *est.*)

- **Anchor (projections, not hidden-run measurements):** main_v5_C (6 Cfull) ~1.2 h; main_v6 (12 Cfull) ~2.2 h.
  So one C member costs about **10 min of wall time on 2xT4**, plus ~0.2 h of fixed decode, load and write.
  One C member per study is 5 views x 24 triplets x 13.1 GMACs = **1.57 TMAC**.
- **B member per study:** 5 views x 16 triplets x ~30 GMACs = **2.4 TMAC**, i.e. **1.53 C-units, *est.* ~15 min** per
  member for the hidden set. ViT-B/16 cost scales with tokens: 23.6 GMACs at 256 px (261 tokens) x 329/261, plus
  attention, gives ~30 GMACs at 288.
  - Assumption: equal MAC efficiency. ViT matmuls usually use T4 tensor cores better than ConvNeXt's memory-bound
    depthwise convolutions, so this is the conservative side.
  - Memory: B's uint8 inputs are ~20 MB per study, against A's ~53 MB.

| Kernel | A | B (4 members) | A+B | x 1.25 | vs 6 h |
|---|---|---|---|---|---|
| main_v5_C + B4 | 1.2 h | 1.0 h | 2.2 h | **2.75 h** | ok |
| main_v6 + B4 | 2.2 h | 1.0 h | 3.2 h | **4.0 h** | ok |
| ceiling for B | - | main_v6: <= 2.6 h (<= 10 members); main_v5_C: <= 3.6 h | | | |

**Pre-declared B membership: 4 full-data seeds, no z-TTA** (z-TTA was measured worthless, §4.5). Two measurements are
required before release:
1. `bench_infer.py` on 200 studies on an A6000, multiplied by the empirical stage-4 factor 2.2.
2. The fp16-vs-fp32 check on the same 200 studies: gap < 0.003 AUC (§2.4 rule). A DINOv3 ViT in fp16 is the main
   unknown here.

If the measured B cost is more than 1.5x the estimate, drop K_infer to 12 before dropping members.

---

## 3. Pooling and fusion

Reuse `model.HEAD_LABEL` (`per_label_attn`) with `slot_prior_strength: 0` and `label_attn_hidden 256`,
`label_attn_drop 0.2`. It is the §4.5 triage winner "without the anatomical prior": 5 folds, stable, a tie with C_p
as a single model, i.e. a proven head that differs from A's.

It keeps the per-slot gated summaries, the learned `[MISSING]` vectors, per-label presence weights, and per-slot aux
heads (aux_weight 0.3 on the 12 keys). The token bag is S + S*K = 5 + 60 in training and 5 + 80 at inference. The
deep `per_label_xfmr` is rejected: one fold diverged in §4.5.

**No new head code.** The only model change is the label width (§8 item 2).

---

## 4. Targets

### 4.1 The 12 keys: `v1_1_gemma3` (shared table), and why label-source independence is not bought

Three measured facts decide this:
1. Arms trained on the **same** label table were diverse before pseudo-labelling: rho(A,E) 0.73, rho(A,C) 0.81, and
   A+E+C gave +0.0126. Only the **shared teacher** collapsed them to rho 0.925 (§4.5). The label table is not the
   correlation source; the teacher is.
2. Extractors share their blind spots. The Claude pilot's lowest correlations were on the same keys for all three
   voters, and a 4th voter's marginal value was +0.0043, below the bar.
3. Any single new-lineage voter scores ~0.85 gold against 0.866 for the 4-voter merge, so a B trained on it would be
   weaker and would dilute the merge. E already showed that weak members dilute.

A new-lineage 12-key pass (Mistral-Small-3.1 is configured, but its Greek/Bulgarian coverage is thin) would cost
~2-4 GPU-h plus a gold check, for a predicted loss. **Decision: the 12 keys use `data/targets/v1_1_gemma3`**
(round 0, fold-local EB calibration). The label table is the one thing B shares with A beyond the evaluation
machinery, and this is stated as a known limitation.

### 4.2 Auxiliary findings: a new aux-only LLM pass (24 keys)

- **Extractor:** Qwen3-32B-AWQ (proven: 100 % parse, prefix-caching layout of V1, `run_screen.sh` hygiene).
  - It uses a **new aux-only schema and prompt**, so none of our existing Qwen/Gemma outputs are reused. The
    independence comes from the target space, which neither pipeline has seen, not from model lineage.
  - Zero-shot + `guided_json`, plus 2 English few-shots written by the developer from non-gold reports
    (`assert_no_gold`).
  - Per key: `state` in {affirmed, hedged, negated, absent} and `evidence` (<= 60 chars, only when affirmed or hedged).
  - *Est.* 2-3 GPU-h. It needs a GPU with >= 30 GB free, so it runs before B training starts on that GPU.
  - Fallback if the co-tenant leaves < 30 GB: `Qwen3-14B` AWQ.
- **Keys (24):**

  | Group | Keys |
  |---|---|
  | Ligaments / tendons | `pcl_tear`, `lcl_plc_injury`, `patellar_tendon`, `quadriceps_tendon`, `acl_graft` (post-op reconstruction), `acl_mucoid` |
  | Cartilage | `cart_medial_tf`, `cart_lateral_tf`, `cart_patella`, `cart_trochlea` |
  | Marrow oedema | `oedema_medial` (MFC/MTP), `oedema_lateral` (LFC/LTP), `oedema_patella` |
  | Meniscus | `men_extrusion`, `men_root_tear`, `men_degeneration` (intrasubstance signal, no tear), `parameniscal_cyst` |
  | Other | `hoffa_oedema`, `plica`, `loose_body`, `osteophytes`, `subchondral_cyst`, `ganglion_cyst`, `normal_study` |

  These are the report's proxies for OA (cartilage, osteophytes), contusion (oedema location) and synovitis
  (Hoffa, plica), which is where A's error concentrates.
- **Aux targets** (no gold, so no calibration):
  - affirmed -> 0.90, weight 1.0; hedged -> 0.50, weight 0.7; negated -> 0.05, weight 1.0.
  - **absent -> weight 0** (unknown, not negative). Exception: `normal_study` affirmed makes every other aux key 0.05
    at weight 0.5.
  - Parse failure -> all aux weights 0.
  - Keep a key only if it has >= 40 affirmed among the 4,349 non-gold rows; drop it otherwise, and record which.
- **Quality check** in place of gold: evidence quote found in the report for >= 95 % of affirmed/hedged cells
  (`evidence_ok` logic), plus a 40-report spot check across en/es/tr. A key with > 20 % wrong cells in the spot
  check is dropped.
- **Aux loss:** `aux_findings_weight 0.3` x masked BCE over the aux columns, i.e. over mentioned cells only. The aux
  logits are never used at inference.

### 4.3 Missing-label loss for the 7 under-reported keys (round 0 only)

The v1_1 targets already give silent cells the fold-local EB **expected-positive** value, e.g. synovitis absent to
~0.29. That is the Cole et al. 2021 correction of "assume negative" on average. What remains is the image-specific
false negative, which is the target of **one-sided Large-Loss Rejection** (Kim et al., CVPR 2022, LL-R).

**Exact form:**
- **Candidates.** In each non-mixup batch, the candidate cells are
  `S+ = {(i,j) : j in UNDER_REPORTED_KEYS, w_ij <= 0.15, sigmoid(z_ij) > t_ij}`.
  - `w_ij <= 0.15` is the existing `pseudo.SILENT_WEIGHT_MAX` convention for silent cells.
  - `sigmoid(z_ij) > t_ij` means the model says "more positive than the report prior".
- **Rejection.** With `l_ij = BCE(z_ij, t_ij)`, `r(e) = ll_reject_max * e / (E-1)` (0 at epoch 0, rising to
  `ll_reject_max = 0.3` at the last epoch), and `n = floor(r(e) * |S+|)`: set `w_ij = 0` for the n cells of S+ with
  the largest `l_ij`. The mask is computed from detached logits.
- **Unchanged.** The masked BCE is otherwise as before. LL-R is skipped on manifold-mixup batches (p 0.3), where the
  targets are blended.
- **Round 1: off** (`ll_reject_max 0`). There, silent cells carry the teacher half at weight 0.5 (the v3 rule).

LL-R only removes gradient from cells that already weigh 0.1, so the risk is near zero and the effect is below
offline resolution. It is part of the recipe, not an ablation.

Entropy-maximisation (Zhou 2022) was rejected: it pushes silent cells toward 0.5, which fights the EB prior.
PU-MLC (dropping negatives) was rejected: negated cells are gold-accurate (§4.1 A/B).

### 4.4 Soft targets per round

| Round | 12 keys | Aux (24) |
|---|---|---|
| r0 (5-fold) | `v1_1_gemma3/fold{k}` as is (soft, EB-calibrated, voter-disagreement weights) + LL-R | §4.2 |
| r1 (full data) | Labels: v1_1_gemma3 recipe calibrated on all 58 gold (the `qa/full/make_targets_full.py` recipe); then `pseudo.mix_with_teacher` with 0.5 label + 0.5 **B-teacher** on every cell of every training row; silent under-reported weight 0.5 (`pseudo.silent_weight_targets`) | Label-only, same as r0 |

---

## 5. Training recipe (all rounds)

| Item | Value |
|---|---|
| Optimiser | AdamW, betas (0.9, 0.999), wd 0.05; no wd on norms, biases, reg/cls tokens (timm `param_groups_layer_decay` no-wd set) |
| LR | backbone top block **5e-5**; **LLRD 0.75** over 12 blocks (patch embed and tokens get the lowest scale, ~1.6e-6); heads 1e-3 (the value per_label_attn was measured stable at) |
| Schedule | cosine; **warmup 1.5 epochs** (the §4.5 divergence hit the first epoch at peak lr after a 1-epoch warmup); 8 epochs (the §4.3 schedule) |
| Regularisation | drop_path 0.1, head dropout 0.1, grad clip **1.0**, EMA 0.999 per step (saved weights) |
| Batch / precision | 4 studies x 5 views x 12 triplets = 240 images; bf16 autocast; grad checkpointing on; `compile_backbone: false` unless the smoke shows >= +20 % img/s with no recompiles |
| Augmentation | unchanged shared code: crop jitter (per view, scaled), rot +-8 deg, b/c +-3 %, sibling swap 0.3, slot dropout 0.1, manifold mixup 0.3 (alpha 0.4); no in-plane flips |
| Selection | `select_epoch: last` (mandatory for full runs) |
| Divergence rule | kill a run if epoch-1 train loss >= epoch-0 train loss, or if its val weak AUC is < 0.60 after epoch 1; one pre-declared retry with backbone lr 2.5e-5 |
| Train cost *est.* | ~130-170 img/s on an A6000 (FLOP-scaled from C's measured 302 img/s at 13.1 GMACs). Fold (3,480 studies): ~26 min/epoch, **~3.5 h**. Full (4,349): **~4.4 h**. Peak memory *est.* 14-20 GB, which fits beside the co-tenant's 26 GB. |

GPU total, *est.*:

| Item | GPU-h |
|---|---|
| Smokes | 3 |
| Aux LLM pass | 3 |
| Round 0, 2 seeds x 5 folds | 35 |
| Round 1, 4 full seeds | 18 |
| Optional 2 extra seeds | 9 |
| **Total** | **~70** (~1.5-2 days of both GPUs) |

The rest of the calendar is fallback slack.

---

## 6. Round 0 -> round 1 plan and leakage argument

1. **r0**: `b_vitb_dinov3_288_r0`, seeds 2026 and 2126, folds 0-4. Every study's OOF comes from the fold model that
   never trained on it. Gold rows are in every fold's validation set and in no training set (`split_studies`
   excludes `is_gold`, and `train_fold` asserts it).
2. **B-teacher** = `pseudo.teacher_mean([oof_s2026, oof_s2126])` over the 4,407 OOF rows. It is built **only** from run
   dirs matching `runs/b_*`: `make_b_targets.py` refuses any other path, and a test pins this. No A OOF, no A pseudo
   target and no A prediction are read anywhere in B.
3. **r1**: `b_vitb_dinov3_288_r1_full`, seeds 2026, 2126, 2226, 2326, `--fold full`, trained on the 4,349 non-gold rows
   with the §4.4 r1 targets. Evaluation is on gold-58 only (`split_full`). **No 5-fold round-1 students.** Their weak
   OOF would be contaminated by construction (the v3 argument: the teacher value of a fold-j study comes from a model
   trained on fold k's rows), and the merge decision is taken on full-data gold anyway. This saves ~18 GPU-h.

**Leakage, point by point:**
- (a) Gold never enters a B training table (round 0, round 1) or a teacher model's training set. The teacher's gold-row
  predictions are OOF and are never written into a target, because `mix_with_teacher` only touches `train_ids`, which
  exclude gold. So the gold-58 scores of B's full students are held-out predictions.
- (b) Gold touches B only through per-label calibration constants: fold-local in r0, all-58 in r1. A's full-data
  members carry the identical all-58 tilt, so the **paired** comparison is not biased toward either side. The LB
  remains the clean judge.
- (c) The B-teacher never sees the study it labels: each OOF value comes from a model that excluded that study's fold.
  A full-data student then trains on every non-gold row. There is no held-out weak set to leak into, which is why
  no weak/strict metric is quoted for r1.
- (d) Aux labels are extracted from report text for all 4,407 studies. Gold rows are dropped from every aux table,
  and no gold label is used in the aux prompt or mapping.
- (e) No second teacher round on B's own students: the round-2 lesson in §4.5.

---

## 7. Evaluation and merge rule (pre-declared, 2026-09-30)

- **A-final** = main_v5_C's 6 Cfull members, with their held-out gold-58 predictions (`full/oof.parquet`).
  If the A side later ships main_v6, the same rule is re-run against it; nothing else changes.
- **Merge** = `0.5 x mean(A members) + 0.5 x mean(B r1 seeds 2026-2326)`, equal weight per pipeline and per member
  within a pipeline. **Never tuned: no per-label weights, no stacking** (stacking lost in §4.5).
- **Primary test (G2):** paired bootstrap (2,000, `metrics.bootstrap_subset_auc` + `paired_delta`) of gold-58 macro AUC,
  Merge vs A-final.
  - **Accept for an LB probe iff delta >= +0.004 and mean B-seed gold >= 0.900.** Report the CI and P, but do not gate
    on P (paired ensemble noise is +-0.007-0.009).
  - Also report rho(B, A) per label (not a gate, §4.1 note) and per-label deltas (not acted on).
- **One LB probe (G3):** the merged kernel, versioned together with its dataset.
  - **Ship as final-selection slot 1 iff LB(merge) >= LB(main_v5_C) - 0.001, i.e. >= 0.947** (paired LB noise
    ~0.002-0.003, 3-decimal display). Slot 2 stays the best A-only entry.
  - If the merge fails, B is dropped. No second B variant is probed.

**Kill gates:**

| Gate | When | Pass condition | On fail |
|---|---|---|---|
| **GL** | 10-01 | rules 5.a.4 + external-model clause re-read from the Kaggle rules text | DINOv2 reg4 fallback (Apache) |
| **G0** | 10-02 18:00 | 200-study / 2-epoch smoke: train loss falls epoch 0->1 and ends at or below C's smoke curve; >= 100 img/s; peak <= 22 GB; bench fp16-vs-fp32 gap < 0.003 on a smoke member; Kaggle image timm >= 1.0.20 or a timm wheel installs `--no-index` | ConvNeXt-S DINOv3 at 384 |
| **G1** | 10-04 12:00 (seed 2026 round-0 OOF) | gold-58 >= 0.870 (C round-0 0.888 - 0.018) **and** paired gold of mean(C_r0, B_r0) - C_r0 >= 0 **and** mean per-label rho(B_r0, C_r0) <= 0.90 | fallback lane (§9) if >= 7 days remain, else stop B |
| **G2** | 10-05 18:00 | §7 primary test | stop B |
| **G3** | LB result by 10-08, hard latest 10-12 | §7 LB rule | B out; slot 1 = A |

**Freeze 10-15:** nothing new after it. Extra B seeds of the frozen recipe (averaged, never picked) may still be added
until 10-20 if runtime allows, followed by a kernel-vs-local re-verification.

---

## 8. Build list for the developer (estimates in eng-days, *est.*)

All config additions default to "off", so every existing config and bundle loads unchanged.
Release hazard (§4.1): new keys make new bundles crash the old kernel. B ships only in a combined dataset + kernel
release built in a worktree.

1. **`config.py`** (0.1 d)
   - `ModelConfig.aux_keys: tuple[str, ...] = ()` and `ModelConfig.aux_findings_weight: float = 0.3`. These go in
     ModelConfig so the bundle records them.
   - `TrainConfig.layer_decay: float = 0.0` (0 = the current two-group optimiser) and
     `TrainConfig.ll_reject_max: float = 0.0`.
2. **`model.py`** (0.25 d)
   - `KneeMIL` uses `n_labels = len(LABELS) + len(cfg.aux_keys)`. `fuse` returns the full width.
   - **`forward` returns `logits[:, :12]` and `aux[..., :12]`**, so `infer._Consumer.predict`,
     `train.predict_studies_cached` and the OOF writer are untouched.
   - `prior_matrix` already returns zeros when `n_labels != 12` (B uses strength 0 anyway).
   - `build_model`/`load_member`/`load_bundle` need no change, because the width comes from `ModelConfig`.
3. **`data.py`** (0.25 d): when `aux_keys` is set, `StudyDataset` appends the `{aux}_target` / `{aux}_weight` columns
   to target/weight (width 12+N). A row without aux columns raises. `synthetic_targets` gains the aux columns for tests.
4. **`train.py`** (0.5 d)
   - `build_optimizer`: with `layer_decay > 0`, `timm.optim.param_groups_layer_decay(model.backbone, weight_decay,
     layer_decay=...)`, each group's lr = `lr_backbone x lr_scale`, heads group as today. `LambdaLR` scales every
     group, so the schedule is unchanged.
   - Loss:
     `masked_bce(logits[:, :12], ...) + aux_weight x masked_bce(aux[..., :12], ...) + aux_findings_weight x
     masked_bce(logits[:, 12:], ...)`.
   - `large_loss_reject(logits, target, weight, epoch, cfg)` exactly as in §4.3, skipped on mixup batches.
5. **Aux labels** (1.0 d incl. the LLM run and spot check)
   - `src/rsna_kad/labels/aux.py`: `AUX_KEYS`, the JSON schema, the aux prompt builder (reusing `prompts.py`'s
     few-shot and `assert_no_gold` machinery), and `aux_targets(table) -> DataFrame` implementing §4.2.
   - `scripts/extract_labels_llm.py --schema aux` and `configs/llm_qwen3_32b_awq_aux.yaml`, producing
     `data/labels/aux_qwen_v1.parquet` + a stats json (parse rate, evidence rate, affirmed count per key, dropped keys).
   - Must run through `run_screen.sh` (the HOME/XDG and `VLLM_NO_USAGE_STATS` incident, §4.1).
6. **`scripts/make_b_targets.py`** (0.5 d)
   - `r0`: `v1_1_gemma3/fold{k}` joined with the aux targets, written to `data/targets/b_r0/fold{k}.parquet` +
     `manifest.json` (labels path, aux path, md5s).
   - `r1-full`: the full calibration recipe (import it from `qa/full/make_targets_full.py`, moved into a function)
     -> `teacher_mean` over `runs/b_*` OOFs only -> `silent_weight_targets` + `mix_with_teacher` -> join aux ->
     `data/targets/b_r1_full/full.parquet`. The manifest records the teacher run dirs and md5s.
   - Asserts: 4,349 rows, no gold, values in [0,1], teacher paths match `^runs/b_`.
7. **Export / kernel** (0.75 d)
   - (a) `export_kaggle.py --pipeline A=<runs...> --pipeline B=<runs...>` writes member weights
     `0.5/n_A` and `0.5/n_B`. This is the proper fix of the §4.5 weight bug; `finalize_ensemble.py` is retired.
   - (b) The fallback prior comes from 4,407-row OOF means (fix of the §4.5 prior bug), using B's r0 OOF and A's
     main_v4 prior.
   - (c) Ship `LICENSES/DINOv3_LICENSE.md` + `NOTICE` whenever any member backbone contains `dinov3`.
   - (d) Notebook prelude: if `timm.__version__ < 1.0.20`, pip-install the shipped timm wheel `--no-index` before
     import. Then re-run A's kernel-vs-local check, since A would run under the new timm too.
   - (e) Rebuild the notebook; `kaggle_export.INLINE_MODULES` is unchanged.
   - (f) On the server: check `pip show timm` in the shared venv. **Do not upgrade the shared venv while A chains
     run.** If it is < 1.0.20, install timm into a B-only `--target` dir on `PYTHONPATH` for B jobs.
8. **Configs** (0.1 d)
   - `configs/b_smoke_vitb_288.yaml` and `configs/b_vitb_dinov3_288_r0.yaml`
     (backbone `vit_base_patch16_dinov3.lvd1689m`, img_px 288, the §2 slots/views, k 12/16, head `per_label_attn`,
     slot_prior_strength 0, aux_keys, layer_decay 0.75, ll_reject_max 0.3, lr_backbone 5e-5, lr_heads 1e-3,
     warmup 1.5, grad_clip 1.0, targets `b_r0`).
   - `configs/b_vitb_dinov3_288_r1_full.yaml` (same, targets `b_r1_full`, ll_reject_max 0).
   - Fallbacks: `configs/b_convnexts_dinov3_384_r0.yaml` and `configs/b_vitb14reg_dinov2_280_r0.yaml`.
9. **Tests** (0.5 d, alongside), in `tests/test_pipeline_b.py` plus extensions:
   - Old configs and old bundle metas load, and the default model is bit-identical (parameter set and RNG order).
   - Width-14 model: `forward` gives (B,12), `fuse` gives (B,14).
   - Dataset aux columns.
   - LLRD group lrs are monotone and heads are unscaled.
   - LL-R: only silent under-reported cells with p > t are rejected; rate 0 at epoch 0; off on mixup batches; count =
     floor(r x |S+|).
   - `make_b_targets`: refuses non-`b_` teacher paths, no gold rows, fold-k contract.
   - Export: pipeline weights sum to 1 with 0.5 per pipeline; licence file present.
   - **Train/infer parity at img_px 288 with the 110 mm and 90 mm views** (byte-exact, extending the existing view-crop
     parity test).
   - A random-init `vit_small_patch16_dinov3` KneeMIL goes through `members_from_ensemble` + `predict_studies` on the
     synthetic fixtures (CPU).
10. **Release rehearsal** (0.5 d): dry run in `.worktrees/release_b`, then 200-study local timing, fp16/fp32 check, the
    3 public studies, dataset file-listing poll (not `dataset-status`), kernel-vs-local diff < 5e-4.

**Total ~4.5 eng-days**, front-loaded on 10-01/02 for items 1-5 and 8. Items 6, 7, 9 and 10 run while round 0 trains.

---

## 9. Calendar 2026-10-01 -> 10-15

Assumption: the ConvNeXt-V2-base seeds finish overnight. If not, B jobs queue behind them via `qa/full/after_cfg.sh`.
**Every GPU queue ends in a named successor** (§4.5 process rule).

| Date | GPU 0 | GPU 1 | Dev / kernel / release |
|---|---|---|---|
| Thu 10-01 | ViT-B@288 smoke (no aux): 200 studies / 2 ep; successor: aux LLM pass (queued) | ConvNeXt-S DINOv3@384 smoke (fallback reference); successor: B smoke with aux + LL-R plumbing | **GL** rules re-read; download timm weights to HF_HOME; server timm check; build items 1-4, 8; aux schema + prompt; tiny Kaggle probe kernel printing timm/torch versions (internet off) |
| Fri 10-02 | aux LLM pass (*est.* 2-3 h) -> aux targets | smoke 2 + `bench_infer` of a smoke member (A6000 s/study, fp16 vs fp32) | aux spot check (40 reports) + key pruning; `make_b_targets r0`; **G0 18:00**; 20:00 launch r0 chains: GPU0 s2026 f0,f2,f4 -> s2126 f1,f3; GPU1 s2026 f1,f3 -> s2126 f0,f2,f4 |
| Sat 10-03 | r0 folds | r0 folds | items 6, 7, 9; the s2026 folds land through the day (never score a live chain) |
| Sun 10-04 | r0 s2126 | r0 s2126 | **G1 12:00** on the s2026 5-fold OOF. Evening: teacher (2 seeds) -> `b_r1_full` targets -> launch r1: GPU0 s2026 -> s2226, GPU1 s2126 -> s2326; successors: s2426 (GPU0), s2526 (GPU1) |
| Mon 10-05 | r1 -> extra seed | r1 -> extra seed | **G2 18:00** (4 pre-declared seeds); export the merged bundle in a worktree; dry run |
| Tue 10-06 | extra seeds finish; then idle-safe filler: none needed (release week) | idem | release: dataset + kernel together; 200-study rehearsal; kernel-vs-local check; **LB probe submitted** |
| Wed 10-07 | - | - | probe scored -> **G3** |
| Thu 10-08 - Mon 10-12 | **fallback lane only if G0/G1/G2 failed**: r0 (1 seed) -> r1 (3 seeds) of the fallback backbone, ~3 days | idem | fallback export + probe by 10-12 at the latest |
| Tue 10-13 - Wed 10-14 | - | - | final-selection proposal to the user (slot 1 = merge if G3 passed, slot 2 = best A-only); inference robustness re-check (compressed DICOMs, time guard) on the final kernel |
| Thu 10-15 | - | - | **freeze** |

If everything passes, B is decided by 10-07 with 8 days of slack. The slack exists for the fallback lane and for
co-tenant slowdowns (x1.5 on every training estimate still lands G2 by 10-06).

---

## 10. Top risks

1. **ViT fine-tuning instability** (precedent: the deep head diverged at peak lr).
   Mitigation: LLRD 0.75, warmup 1.5 epochs, clip 1.0, the divergence rule with one retry, G0 on the train-loss curve.
   Fallback: ConvNeXt-S DINOv3.
2. **B too weak, diluting the merge** (E diluted C after pseudo-labelling). Expected B single gold is 0.900-0.915
   *est.* against C ~0.920. G1 (>= 0.870 round-0) and G2 (mean seed >= 0.900 and merge >= +0.004) catch it, and slot 2
   keeps A.
3. **Not different enough.** B shares the label table, the view geometry (mostly) and the augmentation code, so rho
   could land > 0.90. G1's rho gate is the check. Honest prior: P(G2 and G3 pass) ~35-45 % *est.*
4. **T4 speed / fp16 for DINOv3 on T4.** Runtime is FLOP-estimated only. Mitigation: bench + fp16/fp32 check before
   export, K_infer 16 -> 12 lever, 4 members sized well under the cap. If fp16 fails outright, B cannot ship
   (fp32 on T4 is ~4x slower) and B is killed at G0.
5. **timm version drift.** Kaggle image or server < 1.0.20. Also, RoPE periods are fp32 in timm vs bf16 upstream (card
   note): harmless as long as train and inference use the same timm, which the parity test and the kernel-vs-local
   check pin. A shipped timm wheel also changes A's runtime library, so A is re-verified in the same release.
6. **Licence / rules residual.** The rules text is not re-read today (GL on 10-01). The DINOv3 terms could be amended
   (clause 8). Fine-tuned B weights must be published under the DINOv3 Agreement with a copy attached, never as
   CC-BY-NC. The DINOv2 Apache fallback is pre-declared.
7. **Aux labels noisy or sparse.** No gold for them. Mitigation: evidence check, spot check, key pruning, absent =
   unknown (weight 0), weight 0.3. Worst case they are dead weight; they cannot touch inference.
8. **Shared-GPU contention.** The co-tenant uses up to 26 GB and slows jobs. The LLM pass needs ~30 GB. B training is
   estimated at <= 20 GB. Mitigation: `nvidia-smi` check in `run_screen.sh`, the 14B fallback, resumable `last.pt`.
9. **Release hazard.** New config keys crash old kernels. Dataset and kernel ship together from a worktree, then the
   file-listing poll and the kernel-vs-local diff (§4.1 lessons).
