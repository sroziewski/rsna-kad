# Audit 2026-09-25 - 1. Researcher (state of the art)

Tags: **[V]** verified from a primary source today (Kaggle API pages/discussions/leaderboard, paper page) ·
**[S]** second-hand (participant post or notebook, not reproduced) · **[U]** unverified / our guess.
Sources pulled 2026-09-25 ~23:00 CEST via `kaggle_cli.sh raw` (competitions pages/topics/leaderboard,
kernels list/pull, efficiency-LB kernel output) + web. Scratch copies only; nothing in the repo touched.

## 1. Facts (dated)

**Leaderboard, 2026-09-25 [V]** (4,354 teams on the public LB; `competitions list` says 4,355 teams, our rank 1837):
- Top: 0.959 (rhoskeri), 0.958 (Scott Willis), 0.957 ×3 (takoi & charmq, TomKroumov, Parag & Ian). 21 teams ≥ 0.955,
  64 ≥ 0.950, 189 ≥ 0.944. Top was ~0.957 on 09-18 → **the top barely moved (+0.002); the middle did.**
- Pile-ups = public fork recipes: **355 teams at exactly 0.943**, 200 at 0.942, 356 at 0.941, 290 at 0.936.
- **Our 0.933 = rank ~1832-1857.** Medal lines by count (Kaggle rule for ≥1000 teams, [U] until final team count):
  bronze top 10 % ≈ rank 435 → **0.943 today**; silver top 5 % ≈ 218 → 0.943-0.944; gold 10+0.2 % ≈ 19 → 0.955.
- Efficiency LB (kernel `ryanholbrook/rsna-knee-abnormalities-efficiency-lb`, output 09-25) [V]: **we are rank 478**
  (listed with our 0.931 entry). #1 Scott Willis public 0.958; ranks 1-10 all carry public 0.945-0.958.
  Scott: "single fold, single model ... smaller resnet", 0.949 in ~5 min, own labels via **Gemma 4 locally** ~0.89 gold,
  model ~0.930 on gold [S, topic 735304].

**Host / rules, primary text [V]:**
- **Hosted LLMs are explicitly allowed** (host Po-Hao Chen, topic 733965, 2026-08-09; reply 08-27: "You can use LLM API,
  such as those from OpenAI, to read the reports to generate the labels"; LLM-derived labels/embeddings OK as model
  inputs). **Our architecture §2.1/§4.1 premise "rule 4.b bars hosted APIs" is obsolete since 08-09**; the public
  `pilkwang` (Claude-API) table needs no further ruling.
- **Winner licence = CC-BY-NC 4.0** (rules §5/§6), and rule 5.a.4: pretrained models/data with an incompatible licence
  are exempt from the open-source grant. Host 08-27: NC-licensed external datasets "not prohibited on that basis alone".
  → **CC-BY-NC(-SA) weights such as RadImageNet are not a blocker** (analysis.md §7B over-weighted this). ND licences
  remain a problem for released derivative weights [S, Tucker Arrants 737950]. KneeCoT banned (08-25); OAI allowed only
  if no institutional sign-off (09-21) - community reads that as "no" [S].
- Winners must publish code **and weights** publicly (Prizes page) by 2026-11-05; video required.
- **Efficiency eligibility: the submission "must be among the submissions selected by a team for the Leaderboard
  Prize"** (Efficiency Prize page). With **2 final selections**, the efficiency candidate must be one of the two.
  Formula confirmed: `AUC/(Benchmark−maxAUC) + RuntimeSeconds/32400`, Benchmark = all-0.5 (=0.5). With maxAUC≈0.96:
  **0.01 AUC ≈ 704 s ≈ 11.7 min.** RuntimeSeconds = whole notebook incl. pip, model load, DICOM read (Kaggle staff, 733475).
- Data page: test (~1,300 studies) has **mixed transfer syntaxes (JPEG Lossless, JPEG 2000, Implicit VR)**;
  `Fluid_Sensitive`/`Fat_Suppression` "not necessarily equivalent for every case" (host 08-28: equality in train
  "should not be interpreted as ... must be equal in another dataset"); **prevalence may differ between train, public
  and private.** No host statement on site-held-out vs stratified split (topic 734681 unanswered).
- Labels: image-read by multiple MSK readers with stricter thresholds; image label authoritative (host, 733826).

## 2. Community delta since 09-18

- **Public notebooks 0.941 → 0.943** [V, kernels pulled]: Mattia Angeli "Speedy Raptors" v34 = 20 DINOv2 + A5 +
  RadImageNet heads + 4 Raptor CoAtNet views + **four CoAt readers** (resgated, **D4 depth-zone SWA3**, **Global96**
  96-slice stack, Repair-v1) prob-averaged 25 % each, 0.40 CoAt/0.60 Raptor. A DINOv3-S fold set
  (`tonylica/...dinov3...`, 0.55 DINOv2/0.45 DINOv3) appears in "tony-rank-ensemble". All third-party weights, same
  per-label public-LB-tuned outer map. `kolyaflexcu/rsna-knee-speedy947` claims 0.947 in its slug [U].
- Mattia (742050, 09-19): "no one besides DreadDevelopment and me has released genuinely new models ... changing
  0.55/0.15 to 0.60/0.10 ... is simply overfitting the public LB." Dread: 0.944 in 13 min (CoAtNet + ConvNeXt), 40 models
  in 2 weeks; window count 42→62 +0.003, 48→72 flat [S]; Ziad Ahmed's coverage curve flattens at ~31 windows [S] -
  consistent with our K12→K24 = +0.0033 and "K48 reject".
- **What private top-50 teams credit (all [S], topic 735304 "Best single-model score", 743148):**
  1. **Teacher → pseudo-label → student, blended 0.5 label + 0.5 teacher**: Raymond Yuen 0.938→0.940→**0.949**
     ("multiple weak-label sources (public, api, local) → teacher models → pseudo-labels → final model"; API cost < $5;
     "difference between expensive and cheap [LLM] models minimal"). Archit Konde **single-fold CoAtNet@224 = 0.950**,
     gold OOF 0.930: image model beat its labels on gold, mostly on *silent* cells → filled them with model predictions,
     **mixing** (not replacing, not masking; masking "did nothing"; rounding to 0/1 hurt).
  2. **Resolution is not the driver**: 224-288 px "sweet spot" (Tucker Arrants 0.943@224, tennogh 0.943@288,
     Raymond 384→288 "barely any effect"); simple CNN + simple pooling reaches 0.94+ (Tucker, Tom Aindow).
  3. Label-extraction polish beyond ~0.88-0.89 gold stops moving the LB (Tucker, tennogh, Yann Majewski: labels
     0.86-0.895 gold → "very similar LB"). **Matches our 4.3 finding** (v1.1 +0.015 label → no model gain).
  4. Heavy aug + regularisation; 5-fold + TTA; effnet-b0 teams at 0.94+ with 0.88-AUC labels (Cody_Null) [S].
  5. Novel: **Qwen3.5-2B VLM, LoRA on LM + vision encoder, 384 px, single fold 0.950**, ~3.5 h (NguyenThanhNhan) [S].
  6. RadImageNet encoder lifts synovitis 0.62-0.72 → 0.78 (starkhushi, 737566) [S].
- **Gold reference points** for single models (their OOF on the 58) [S]: Archit 0.930, Scott ~0.930, Salem 0.921.
  **Ours: arm A 0.890, arm C 0.888, A+E+C fold-0 0.914.** Our image models trail the leaders by ~0.03-0.04 on the same
  58 studies (CI ±0.03, so indicative only) - the same order as the LB gap (0.933 vs 0.95-0.96).
- Fast-inference recipe (Prateek Grover, 743374, 09-25, CoAtNet 40→<10 min) [S]: one model replica per GPU (no
  DataParallel); 16-thread DICOM reads (59→372 MB/s from /kaggle/input); raw header parse (0.307→0.058 ms/file);
  `np.frombuffer` on PixelData (1.22→0.33 ms/slice, uncompressed only); strided-subsample percentiles (sort was 60 %
  of decode); **fp16 weights + half inputs, no autocast = 2.08× over autocast on T4**; uint8 to GPU, normalise on device.
- GitHub (pushed ≥ 09-18, 43 repos for "rsna knee"): all early-stage scaffolds; nothing above public-notebook level [V].
- Report-generation-from-image round trip: negative (743148) [S].

## 3. SOTA by lever

| Lever | Evidence | Expected gain for us | Cost | Offline T4 | Licence |
|---|---|---|---|---|---|
| (a) Pseudo-label blend 0.5/0.5 (teacher = our ensemble) | RSNA-2024 lumbar 2nd [V, brendanartley repo]; 3 independent top-50 reports here [S] | **+0.005-0.015 LB [U guess]**, concentrated in synovitis / pf_oa (stage-9: 3,700 silent synovitis cells) | 1 extra training round of the chosen arms (~15-30 GPU-h) | n/a (training) | ours |
| (a) Report-uncertainty-aware smoothing | Rep-GLS, arXiv 2508.02495 (Aug-Nov 2025) [V]: LLM-extracted hedge words → per-sample smoothing rate | ≈ what our hedged/mild state mapping already does; 0 new | 0 | n/a | - |
| (a) Generic noisy-label methods (co-teaching, robust losses, confident learning) | LNMBench, arXiv 2512.09315 (Dec 2025, Pattern Recognition 2026) [V]: LNL methods "degrade substantially under real-world noise" | ≤ 0.003 [U]; our stage 9 already refuted CL | medium | n/a | - |
| (b) Per-finding attention over windows/slots | every 0.94x public CoAt reader ("finding-specific gated spatial head" + per-finding window attention) [V code] | +0.003-0.01 [U]; targets lat_oa/mcl | 1 new arm | yes | ours |
| (b) Depth-zone / wide-span sampling (D4, Global96, 78-96 windows) | Dread/Mattia public arms [S] | ≤ 0.003 given our flat K curve | inference time | yes | ours |
| (b) 2D-encoder + slice transformer, multi-view | 2025 knee papers (eClinicalMedicine 9-abnormality multi-task; MeniMV arXiv 2512.18437) [V titles] | nothing beyond our MIL | - | - | - |
| (c) RadImageNet ResNet-50 member | synovitis 0.78 vs 0.62-0.72 [S]; public stacks weight it 0.55 | +0.002-0.006 macro via synovitis [U] | 1 cheap arm (~8 GPU-h) | yes, fast | CC-BY-NC-SA → OK under 5.a.4 / NC winner licence [V rules; SA nuance U] |
| (c) DINOv3-S/B | in public 0.943 stacks [V]; one team saw v3 < v2 [S] | 0-0.005 [U] | 1 arm | yes | DINOv3 License: commercial allowed, custom, attribution [V meta] |
| (c) MRI-CORE (SAM ViT-B, 6.9 M MRI slices) | arXiv 2506.12186 [V]; 1024 px input, seg-focused | low; too slow at 1024 on T4 | high | marginal | Apache-2.0 [V] |
| (c) Pillar-0 knee fine-tune | Kaggle model `fabotelli/pillar0-knee-finetune`: frozen 0.697 → 0.797 macro, ~0.10 below 2.5-D [S] | ≈0 | - | yes | check model card [U] |
| (c) MARS (Nat. BME 2026, 336 k MRI volumes) / OrthoDiffusion | weights availability [U] / unreproduced (0.883 vs 0.958 claimed) [S] | skip | - | - | - |
| (c) Small VLM (Qwen3.5-2B + LoRA) | single fold 0.950 [S] | unknown; high variance bet | 1-2 days eng. | 2B fp16 fits T4 [U] | Qwen licence [U] |
| (d) Ensembling under tiny noisy val | our §4.2 + 2023 abdomen 1st (equal weights, CV-tuned scaling only) [V repo]; public LB-tuned per-label maps flagged by their own authors | keep equal-weight mean; **no LB tuning** | 0 | - | - |
| (e) Distil ensemble → one fast student | standard KD; pseudo-label student *is* a distilled model | eff-track: student ≈ ensemble −0.005 at 1/10 runtime [U] | shares (a)'s round | yes | ours |
| (e) Inference engineering | 743374 list above; fp16-no-autocast 2.08× [S] | 20→<10 min ≈ +0.009 AUC-equivalent | 1-2 dev days | yes | - |

## 4. Endgame practices of past RSNA winners

- **2024 lumbar 2nd** [V repo]: pseudo-labels as `0.5·loss(labels) + 0.5·loss(pseudo)` in the best entry; confident-learning
  denoising tried only on the last day; 9-rotation TTA; team ensemble of *different pipelines*, not more seeds.
- **2023 abdomen 1st** [V repo]: many seeds per architecture (≥ 12 distinct seeds listed), 4-fold GroupKFold **plus full-data
  models**, slice-level ensemble inside a family and max-aggregation across families; "no major post-processing except
  scaling factors tuned on CV" - never on LB.
- **2024 lumbar 1st / 2022 cervical 1st**: writeups JS-locked; details [U]. Consistent pattern across all four:
  final 2-3 weeks = seeds / full-data retrains / one pseudo-label round of *existing* recipes, not new architectures.
- Selection norm: one CV-best robust ensemble + one diversified/riskier pick. Here Kaggle forces a twist:
  **efficiency eligibility requires the fast model to be one of the two picks** (§1).
- Shake-up risk here: ~900 of top-1,100 public teams run LB-tuned forks of the same weights; the public split is ~390
  studies and prevalence may differ (§1). Past fork-dominated boards reshuffled heavily [S, PC Jimmmy 742327].

## 5. Top 8 research-backed recommendations (27 days; offline resolution ~0.005, public LB ~0.02)

1. **Run one teacher→student pseudo-label round, 0.5 target + 0.5 teacher, on ALL cells (incl. silent ones).**
   Weakness: synovitis under-reporting (12.5 % affirmed vs 46.6 % gold) and our weight-0.1 "absent" cells, plus the
   ~0.03-0.04 gold gap to single models that did exactly this. Teacher = A+E+C OOF ensemble. Stage 9 was rejected on
   weak-OOF, which **by construction penalises departures from report labels** - it cannot see this lever; judge it on
   gold-58 + consensus and one LB probe (accept only ≥ +0.02 LB or gold ≥ +0.02). Evidence: 3 top-50 reports + lumbar-2nd.
2. **Retire the "no hosted API" constraint; add the public `pilkwang` table and optionally one cheap frontier-API pass as
   extra teachers for step 1** (host allowed it 08-09/08-27; < $5). Not for label polish (+0.015 does not move the model,
   our 4.3 and the forum agree) but as diverse teacher inputs, esp. lat_oa / mcl severity cells.
3. **Make the efficiency entry a distilled student and put it in the final two.** Train effnetv2_s@256 (or resnet@224)
   on the ensemble's soft outputs; apply the 743374 I/O list + fp16-no-autocast; target < 10 min. Weakness: eff rank 478,
   0.913 LB; 0.01 AUC = 11.7 min, and without being a final pick it is ineligible.
4. **Add a per-finding attention reader (12 queries over slot×slice, anatomical slot prior) as the next new arm** -
   the one structural feature shared by every 0.94x public CoAt reader. Weakness: lat_oa 0.863 / mcl 0.869 are
   compartment-/edge-specific; arm D's 100 mm coronal view belongs in the same arm (MCL survives only ≥ 100 mm).
5. **Add a cheap RadImageNet-ResNet-50 (or DINOv3-S) member trained on our targets** - licence is OK under CC-BY-NC +
   rule 5.a.4. Weakness: synovitis; community reports +0.06-0.16 synovitis AUC from medical pretraining [S]. Gate on
   gold synovitis + consensus, bundle if < 0.005 macro.
6. **Spend late GPU on full-data retrains + 2nd seed of the best arms (C, then A), not on new ideas after ~10-12.**
   Past winners did seeds/full-data in the last weeks; each fold model sees only 80 %. Guess +0.003-0.006; unmeasurable
   offline, so it is a cost-free "bundle" item under §4.2.
7. **Harden against declared test differences before 10-15:** slot selection must not assume `Fluid_Sensitive ==
   Fat_Suppression`; rehearse with JPEG-LS/J2K/Implicit-VR re-encoded copies of train series; no constant-prediction tie
   blocks for missing slots (ties cost AUC). Weakness: a silent slot mis-route on private data would cost more than any
   modelling gain.
8. **Final selection: pick 1 = equal-weight ensemble (robust, no per-label weights); pick 2 = fast student (efficiency +
   hedge).** Do not copy LB-tuned maps; the 0.943 fork mass is likely to shake down [U]. Consider a team merger before
   **10-15** with a team holding a decorrelated self-trained model (e.g. 743374 author seeks one) - only if weights are
   theirs and releasable.

Guesses above are marked [U]; every "+x" not measured by us is a forecast, not evidence.
