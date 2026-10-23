# Competitor review: `romantamrazov/rsna-knee-dinosaur-v5` (public LB 0.942) — 2026-09-21

Artefacts here: `cells/cell_NN.py` (33 code cells of the inference notebook), `markdown.md`,
`kernels/{rsna-knee-e11-train,rsna-knee-e13-train,rsna-knee-dinosaur-v5-train}/src.py`,
`llm_labels/` (their public label table + labeller), `dino_manifest/manifest.json`. Nothing outside
`qa/competitor/` was touched; no training was run. Ours at the time: LB 0.931.

## 1. Provenance — the 0.942 is a blend of other people's weights

**Zero of the 0.942 graph is trained by the notebook's author.** Every checkpoint is a mounted public
dataset. The author's own trainer (`kernels/rsna-knee-dinosaur-v5-train`, ConvNeXt-tiny + a label-aware
attention head) is listed in `kernel_sources` but **its output is never read**: no cell references
`dinov5_convnext_manifest.json`.

| stage | family | source | trained by | weight in the fusion |
|---|---|---|---|---|
| 1 | 20 × DINOv2-small, 336 px, 6 slots × 12 slices, 130 mm | `pilkwang/rsna-knee-weights` | Pilkwang | base, equal (4 seeds × 5 folds) |
| 2 | A5: 5 folds, timm + `LabelAttentionPool`/`CodexResidualPool`, 336 px, 16 slices | `mattiaangeli/knee-mri-fold-weights` | Mattia Angeli | `a5_w = 0.52` rank-mix |
| 3 | RadImageNet ResNet-50 + 3×5 attention heads + an 88-feature linear calibrator | `marwanmath/...`, `prvsiyan/...`, `antoinegg1/...`, `sofiaanjenje/rsna-knee-e13-train` | four others | `rad_alpha 0.55`, `rad_e13_member_w 0.50`, `rad_second_alpha 0.20`, `rad_cal_w 0.40` |
| 4 | 4 Raptor views from 3 CoAtNet-rmlp-2 SWA checkpoints | `dreaddevelopment/raptor-knee-*` | Dread Development | views 0.60/0.10/0.10/0.20, then per-label `coatnet_w` |
| 4b | 2 CoAt children (resgated-top3, D4 depth-zone SWA3) as child processes | `mattiaangeli/rsna-knee-coat-*` | Mattia Angeli | equal rank mix, `coat_private_alpha = 0.40` |
| 5 | FineSpacing v9 CoAtNet, 80 slices / 78 windows | `dreaddevelopment/raptor-knee-finespacing` | Dread Development | rank residual at scale 0.10 |

Two caveats. (a) `cell_32.py` states "the parent pipeline has already written its exact 0.942
`submission.csv` before this cell runs" and then **overwrites** it with the FineSpacing variant — the
shipped file is *not* the 0.942 artefact but an unscored perturbation of it. (b) Reproducible by us
without third-party weights: **none of it**. The only reusable pieces are ideas, the preprocessing
constants, and one public **label table** (§6).

## 2. Ensembling mechanics — and the LB-tuning verdict

Everything combines in **rank space** (`rsna_rank01`/`rsna_rankpct`, per column), never in probability
space, except the 4 Raptor views which are probability-averaged then ranked once. `TTA_POOL = 'prob'`
in the shipped run; `'logit'` is dormant. Chain: `_combine` (equal rank-mean of the 20 DINO members) →
`_a5_blend_available` ((1−w)·parent + w·a5 on ranks) → `_rad_main` (three rank mixes plus the
calibrator) → Raptor/CoAt → FineSpacing residual. SWA is not done in-kernel; it is baked into
third-party `*_swa.pt` checkpoints.

**Their per-target weights are public-LB-tuned and the author says so** (`cell_00.py:39-46`):

> "Lateral Meniscus at 1.00 discards the DINO, A5 and RadImageNet stages entirely for that column —
> three of four stages, **on the strength of a public split that is a fraction of the test set**. It may
> be real. It is also the likeliest place to give back points privately."

`coatnet_w = {__default__ 0.60, ACL 0.75, Medial Meniscus 0.80, Lateral Meniscus 1.00, Lateral OA 0.75,
Fracture 0.75}`; per-target TTA pooling maps; `_RAD_EXCLUDE = ("Baker's", "Fracture")`. The scalars are a
visible LB ratchet — `a5_w` 0.45→0.52, `rad_alpha` 0.50→0.55, `rad_second_alpha` 0.15→0.20, three
hill-climbs on ~390 studies for +0.001. The 88-feature calibrator (88×12 L1-sparse coefficients over
stage ranks, their differences, group means and protocol counts, gated to 7 labels) carries **no record
of what it was fitted on**. Only the RadImageNet alpha rung is defended with nested GroupKFold.

Under our §2.6/§4.2 this is forbidden and mostly unverifiable: the LB resolves ~0.02, so 0.941→0.942 is
a fifth of the noise. Their own diagnostics cell prints "where a weight fitted to the public split can do
real damage privately". **Read 0.942 as ~0.93–0.94 of real signal plus ~0.005–0.01 of public-split
fitting that will not survive the private split.**

## 3. Preprocessing — near-identical to ours, worse in two places

Stage 1: `CROP_MM = 130.0` (identical to ours), `CACHE_IMG = 336`, 6 slots, `SLICE_BAND (0.2, 0.8)`,
12 slices/slot, uint8, p1/p99 on the selected sub-volume. A5: 336 px, `N_SLICE = 16`, band (0.12, 0.88),
**per-slice** p1/p99. Raptor: `CROP_MM = 140.0`, 336/384 px, span (0.02,0.98), per-slot slice budgets
(Sag-FS 18, Sag 14, Cor-FS 12, Cor 8, Ax 12), p2/p98, then 94 overlapping RGB triplets.
"maxspan/native384/native384dense/depthzone" are four *view* recipes, not different code paths.
Ordering: IPP projected on the slice normal, InstanceNumber fallback.

Differences that matter: they sample a **fractional depth band** and a **fixed per-slot slice budget**
(more sagittal fat-sat than axial) where we take a uniform ≤40-slice subsample and the same K everywhere;
and they sweep far more slice positions at inference (78–94 triplets vs our K=24). We are **better** on
intensity (per-series windowing vs their per-slice, which flickers along z) and on crop handling (we pad
sub-130 mm FOVs; they skip the crop when `want >= min(h,w)`).

## 4. Heads — the one genuinely different idea is per-label attention

- `SlotHead`: 12 learned queries, one per finding, attend over the **6 slots**; an anatomical prior
  (`SLOT_PRIOR_TABLE`, e.g. ACL → sagittal slots, strength 0.55) is *added to the attention logits*
  before the masked softmax.
- `RaptorClassifier`/`_FSClassifier`: `Linear(F,256)-Tanh-Dropout-Linear(256,12)` gives **12 attention
  scores per window**, softmax over windows, per-label read-out. This is our gated-attention MIL with the
  single shared attention replaced by 12 per-label ones, pooling over all slots' slices at once instead
  of per-slot then concat.
- `LabelAttentionPool` / `TokenXAttnPool` / `CodexResidualPool` / `_GatedDelta`: the same idea at three
  strengths — segment-softmax label attention with a learned `slot_bias[label, slot]`; 12 label queries
  cross-attending over *patch tokens*; and a mean/max+presence baseline plus a **gated residual delta**
  so attention can only perturb a safe baseline.
- `SlotDepthMixer` = learned plane/contrast-conditioned 5-tap blur along z (α ≤ 0.25). `DepthCompress` =
  gated 1×1 convs compressing 16 slices to 3 channels (a learned alternative to our fixed `[i-1,i,i+1]`).
  `_RadHead` = 12 label queries over (plane × slice) tokens, fused as `[attn, mean, |attn−mean|, attn*mean]`.

Transferable core: **one attention distribution per label, plus an anatomical slot prior on its logits.**

## 5. Flips — they canonicalise, they do not augment; sagittal is a depth reversal

No D4/dihedral TTA anywhere. `normalise_laterality` (`cell_15.py`, applied at **inference**): right knee
→ `torch.flip(img, dims=[-1])` for Coronal/Axial, else `torch.flip(img, dims=[0])` — dim 0 of
`(n_slice,H,W)` is the **slice axis**. That answers our §2.3 open question: medial↔lateral in a sagittal
stack lives on the *through-plane* axis, so the sagittal laterality flip is a **slice-order reversal**,
not an in-plane mirror — **our ban on the in-plane sagittal flip stands** (it would swap
anterior/posterior). Side comes from the `Laterality` tag, else from the FOV-centre x via IPP+IOP+
PixelSpacing — the same construction as our `side_geom`. A5 is inconsistent (flips coronal/axial only,
leaves sagittal uncanonicalised). Raptor does not flip; its `maxspan-v5-reverse` view reverses the
triplet channel order (a depth-direction TTA at weight 0.10). So: flipping is safe **as canonicalisation
conditioned on an estimated side**, never as a blind label-preserving augmentation.

## 6. Training side (from their kernels)

Folds: `md5(report text) % 5` + `GroupKFold` on report groups. **Their own EDA** (`cell_06.py:52-57`)
calls this out: "does nothing about the scanner… Manufacturer + model + matrix size + pixel spacing is a
serviceable site key… Grouping on it is what makes an offline number mean something." Our site-grouped
folds are strictly better; their offline numbers are inflated, which is why they fell back on the LB.
Optimiser: AdamW, `LR_BACKBONE 8e-6` / `LR_HEAD 1e-3`, wd 0.02, **OneCycleLR** (pct_start 0.15),
10 epochs, fp16, DINOv2 frozen except the last 6 blocks. **No EMA and no SWA in their own code** — they
keep the best epoch on the holdout, which our stage-3 follow-up measured as ~+0.008 of optimism. Loss:
weighted BCE on soft targets, no focal; the fork's trainer adds `0.88·soft_bce + 0.12·pair_rank_loss`
and selects the best epoch on a 5-fold split of the **58 gold rows only**.

Labels: a multilingual lexicon overridden by `pilkwang/rsna-knee-llm-labels` (`report_labels_v2.csv`,
score/conf/verdict per finding), weight `0.25 + 0.75·conf`, gold weight 3.0. **Measured here** (57 shared
gold rows): their table scores **gold-58 macro AUC 0.870** vs our `llm_v1` **0.850**; the rank-average of
the two scores **0.877**, paired bootstrap **+0.0270 [+0.0148, +0.0381], P = 1.000** over ours alone
(theirs alone +0.0205, P = 0.986). Per label: ACL 0.882→0.997, Contusion 0.712→0.788, Fracture
0.821→0.851; synovitis equally bad in both (0.69). Per-label Pearson between the extractors is 0.879
(lowest Fracture 0.717, Contusion 0.745) — genuinely diverse voters. Their labels came from a **hosted
API** (`api_labeler.py`: `anthropic`, `model="claude-opus-5"`, JSON-schema constrained) — we may not call
one (§2.1, rule 4.b), but the *published CSV* is ordinary public external data. **Get a host ruling
before relying on it in the final entry.**

## 7. Recommendations (bar = §4.2: ≥ 0.010 with P ≥ 0.95, else bundle or reject)

**(A) Adopt in our own pipeline.**

| # | change | expected gain | cost | clears the bar? |
|---|---|---|---|---|
| A1 | **targets v1.1: a 4th voter** — either the public `pilkwang` table (rank-averaged with our state-mapped score, `__conf` folded into the weight) or **our own second local LLM family** for clean provenance | **+0.027 gold-58 measured** for the public table; stage 7 showed an extractor jump of this size was worth **+0.0265 OOF-weak** | ~1 CPU-h + ~27 GPU-h to retrain A and C (plus ~2 GPU-h if we run our own second extractor) | **YES** — the only single change that clearly does |
| A2 | **per-label attention head**: 12 queries over the (slot × slice) bag + an anatomical prior on the attention logits | structural difference vs all three of their arms; plausible 0.005–0.015 | 2.5 GPU-h smoke, then 12–15 GPU-h | only if the fold-0 paired delta reaches 0.010 |
| A3 | **sagittal laterality canonicalisation by slice-order reversal** where `side_conf ≥ 0.9` | answers §2.3 with working code; unlocks flip augmentation on ~50 % of studies | folded into A2 | bundle with A2 |
| A4 | **per-target max/top-2 pooling over z-offset TTA** for focal findings | their EDA gives the mechanism: a tear on two slices falls between samples and mean-pooling dilutes it; +0.003–0.008 | ~0, inference only | bundle |
| A5 | **pairwise rank loss** `0.88·BCE + 0.12·softplus(−(z_pos − z_neg))` | metric is AUC; < 0.005 expected | ~0 | bundle |
| A6 | denser inference sampling (K24 → K48) | our own sweep says we are on the flat part (K12→K24 = +0.0033) | 2× kernel GPU | **reject** |

**(B) Only works by blending third-party weights.** Stages 1–5 *are* the 0.942. Public pretrained models
and datasets are allowed, but the winners' obligation requires open-sourcing the final model, and a model
that is 100 % other people's checkpoints is not ours to open-source or defend — we could not retrain it,
and the source datasets could be deleted at any time. The RadImageNet encoder is additionally
**CC-BY-NC-SA-4.0** by their own audit line, a licence problem for a prize-eligible entry.
**Recommendation: do not build on third-party checkpoints.** A1 is the exception — it is data, not
weights, and we re-derive it into our own targets.

**(C) Already as good or better.** 130 mm constant-mm crop; IPP/InstanceNumber ordering; uint8 cache;
**per-series** p1/p99 windowing (theirs is per-slice — z-flicker); FOV padding below 130 mm; equal-weight
mean-of-sigmoids; **site-grouped folds** (theirs are report-md5, and their own EDA says so); EMA (they
have neither EMA nor SWA in their own trainers); fingerprint-gated checkpoint loading; 384 px vs 336.

**(D) Do not copy.** Per-label outer weights; the LB-ratcheted scalars; the opaque 88-feature calibrator
with no recorded fit set; the FineSpacing residual injected at an untested scale over the scored anchor;
best-epoch selection on 58 gold rows; report-md5 folds.
