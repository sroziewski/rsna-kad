# SOTA literature scan for pipeline B - 2026-09-30 (researcher)

Scope: journals, conferences, arXiv only (Kaggle material is in `qa/audit_2026-09-25/1_researcher.md` and is not
repeated). Context read: `docs/architecture.md` §2 and §4.1-§4.5.

Tags: **[V]** verified today from the primary page (arXiv abstract/HTML, GitHub, HF card, journal page) ·
**[K]** from my background knowledge of a well-known paper, not re-opened today (venue/year are reliable,
fine details less so) · **[U]** unverified / guess. Every "+x" for us is a **guess** unless it cites our own
measurement.

Our starting point in one paragraph: 2.5-D gated-attention MIL over slice triplets, ConvNeXt-V2-tiny FCMAE,
5 views @384, soft targets from 4 report voters with fold-local EB calibration, one pseudo-label round
(0.5 label + 0.5 teacher) = the biggest gain; ~0.920 gold-58 per model, 0.948 public LB. Arms converged
(rho >= 0.92). Seeds, backbone size, heads, stacking, a second PL round, z-TTA: all ties.

---

## 1. Facts per research question

### Q1. Image-report multimodal learning

**What the 2025-2026 evidence says, most relevant first**

1. **Bergström, Yue, Johansson, "When are radiology reports useful for training medical image classifiers?"**
   arXiv 2510.24385 (Oct 2025) [V] - the only systematic study of our exact question. Compares SSL (RAD-DINO,
   Medical MAE, C2L), image-text alignment (BioViL-T, GLoRIA, BiomedCLIP) and MRM (masked image+text) pretraining,
   plus **generalized distillation** (teacher sees image+report, image-only student; LUPI) at fine-tuning.
   Findings: report-aligned pretraining helps **only when the label is well represented in the text**, and can
   hurt otherwise; **"distillation works poorly if the text is too predictive"** of the label (diagnostic
   MIMIC labels: no gain over image-only), while it helps prognostic, text-weak labels in small-data regimes.
   No code link on the page.
   *Implication for us:* our labels *are* derived from the text, so a text-seeing teacher would largely
   reproduce the report labels we already train on. LUPI-style distillation is predicted to add little; and
   report-alignment pretraining is predicted to help the well-reported labels (ACL, menisci, effusion, MCL) and
   **not** the under-reported ones (synovitis, fracture, contusion, OA) where our loss sits.
2. **Silva-Rodríguez, Dolz, Ben Ayed, "A Reality Check of Vision-Language Pre-training in Radiology: Have We
   Progressed Using Text?"** IPMI 2025, arXiv 2504.05227 [V]. **Unimodal supervised pretraining with fine-grained
   labels is highly competitive with, and better at integrating heterogeneous sources than, CLIP-style VLP**
   when image-text data are limited and noisy; proposes DLILP (labels first, text as a secondary noisy signal).
   Code `github.com/jusiro/DLILP` (page states CC BY-NC-ND 4.0 - ND, so do not ship derived weights; the idea is
   free to re-implement).
3. **MedMAP / MedMoM-MRI3D**, arXiv 2602.23652 (2026) [V]: 3D MRI VLP on 7,392 volume-report pairs; design detail
   that matters for us: **text encoder frozen, vision encoder fine-tuned**. Code `github.com/RomantiDr/MedMAP`,
   licence not stated. Knee content not stated.
4. **fVLM**, ICLR 2025, arXiv 2501.14548 [V]: anatomy-level contrastive alignment (organ region <-> the report
   sentences about that organ) on 69k CT patients; the gain over global CLIP comes from **local, per-structure
   alignment plus false-negative handling** (most per-anatomy pairs are "normal"). Code `alibaba-damo-academy/fvlm`.
   Needs segmentation masks per anatomy; 69k scale.
5. **Decipher-MR** (npj Digit. Med. 2026, arXiv 2509.21249) [V]: 3D MRI VLP, 200k series / 22k studies, SSL +
   report supervision; **weights not shared**. **Mr3D-VL** arXiv 2608.12689 and **PRISM** (336k MRI, arXiv
   2508.07165) [V]: large MRI FMs; weights not stated.
6. **Merlin** (Nature 2026) [V]: 3D CT VLM trained on CT + EHR codes + reports; weights on HF (`stanfordmimi/Merlin`).
   CT abdomen only - wrong modality/anatomy for us; cited as evidence that **structured codes + report text
   together beat either alone** at 15k-study scale.
7. Classic 2D chest-X-ray family [K]: ConVIRT (MLHC 2022), GLoRIA (ICCV 2021), MedCLIP (EMNLP 2022, decoupled
   image/text + semantic-matching loss to fix false negatives - explicitly aimed at *small* paired data),
   BioViL (ECCV 2022) / BioViL-T (CVPR 2023), MGCA (NeurIPS 2022), PRIOR (ICCV 2023), MedKLIP (ICCV 2023,
   extracts entity triplets from reports and supervises via entity queries), KAD (Nat. Commun. 2023, MIT,
   knowledge-enhanced entity queries) [V repo licence]. All pretrain on 200k+ English CXR pairs. "Language over
   labels" (AACL-SRW 2022) [V title] claims contrastive supervision beats label supervision on CXR, but on
   hundreds of thousands of pairs; CXR-CLIP (MICCAI 2023) [V] shows mixing label-prompts with reports helps.
8. **Report generation as an auxiliary task**: no paper found that shows a *classification* gain from an
   auxiliary report decoder at our scale; spine/liver/CT report-generation papers (2025-2026) optimise text
   metrics. Combined with the Kaggle-forum negative round-trip (audit §2) -> **no evidence it helps; skip**.
9. **Multilingual reports**: no 2024-2026 paper found on image-text alignment with reports in many languages.
   Nearest: PadChest (Spanish) and PadChest-GR (bilingual, NEJM AI 2025) [V]. Practical routes are
   (a) a multilingual frozen text encoder (BGE-M3 or multilingual-e5-large, both MIT [K]) or (b) translate all
   4,407 reports to English with our local Qwen3-32B (1-2 GPU-h) and use an English biomedical encoder.
   Our LLM extractor already normalises the 9 languages into a structured schema, which is the strongest form of
   (b) - see idea R2.

**Answer to "which gives the most independent signal?"** The literature ranks it: (i) *fine-grained structured
supervision beyond the 12 targets* (Reality Check, MedKLIP/KAD entity supervision, Merlin's codes) > (ii) local
report-sentence alignment (fVLM; needs anatomy masks, large data) > (iii) global CLIP-style alignment (needs far
more than 4.4k pairs; helps well-reported labels only) > (iv) LUPI distillation from a text-seeing teacher
(Bergström: fails when text predicts the label - our case). None of (ii)-(iv) has evidence on 4.4k MRI studies.

### Q2. Pretrained models usable offline on a T4

| Model | What / data | Weights | Licence | Fit for us |
|---|---|---|---|---|
| **DINOv3** ViT-S/B/L/16 and distilled **ConvNeXt-T/S/B/L** (`timm/convnext_*.dinov3_lvd1689m`) | SSL distillation from ViT-7B, 1.7 B web images (2025) | yes, timm/HF [V] | DINOv3 License (custom, commercial use allowed; not ND) [V card; exact redistribution terms: check] | **Best drop-in "different pretraining objective"**: same timm API, ConvNeXt variant fits our code unchanged; ViT variant is a different family |
| DINOv2 ViT-S/B/L | SSL, LVD-142M | yes | Apache-2.0 [K] | used in public 0.943 stacks (audit) |
| **RadImageNet** R50/DenseNet/IncV3 | supervised on 1.35 M CT/MRI/US slices incl. MSK MRI (Mei et al., Radiol AI 2022) [K] | yes | CC-BY-NC-SA per audit [V rules; SA nuance U] - OK under rule 5.a.4 | Medical supervised pretraining; community reports synovitis lift [S, audit]; 224 px, old CNN family |
| **MedImageInsight** (Codella et al., arXiv 2410.06542) | 360 M image encoder trained on image-text-label across X-ray/CT/**MRI**/... | yes | **MIT** [V] | Text-aligned medical encoder, permissive; 360 M is heavy for 5 views x 24 triplets on T4 [U] |
| **MedSigLIP-448** (Google HAI-DEF, 2025) | 400 M SigLIP vision tower, trained incl. **MRI slices** + text | yes | HAI-DEF terms (custom, not ND) [V] | Text-aligned; ~8x the FLOPs of ConvNeXt-V2-tiny@384 [U]; feasible only as 1-3 members |
| BiomedCLIP ViT-B/16 (PMC-15M) | figure-caption CLIP | yes | MIT [K] | 224 px, literature figures - weak on MRI texture [U] |
| **MRI-CORE** (arXiv 2506.12186) | SAM ViT-B on Duke-110K (116k volumes, 6.1 M slices, single site, **no OAI/fastMRI**) | yes (`mazurowski-lab/mri_foundation`) | Apache-2.0 [V] | 1024 px input, segmentation-oriented, no classification AUCs reported [V] - too slow at native res |
| SAM-Med3D(-turbo) | 3D SAM on SA-Med3D-140K | yes | Apache-2.0 [V] | 128^3 segmentation encoder; no evidence as classifier init |
| MedicalNet (Chen et al. 2019) | 3D ResNets on 3DSeg-8 | yes | MIT [U] | the only 3D ResNet init with MRI; old, small data |
| **OrthoDiffusion** (arXiv 2602.20752) | 3 orientation-specific 3D diffusion backbones, 15,948 **private** knee MRIs (PKU) | HF `lanstat0123/orthodiffusion` [V] | **MIT** [V] | The only knee-specific MRI FM with permissive weights, not OAI. Audit [S] says a Kaggle attempt reproduced 0.883 vs 0.958 claimed. Diffusion encoder: slow, unusual API |
| **KneePreM** (arXiv 2609.31461, 2026-09-25) | 3D U-Net MAE on 19,011 **OAI** series | not stated [V] | - | **OAI-trained - flag, avoid** (host: OAI only without institutional sign-off). Weak AUCs (0.72-0.74) |
| 3DReasonKnee (PSB 2026, arXiv 2510.20967) | 7,970 knee MRI grounded-reasoning dataset (Rajpurkar lab) | dataset on HF, CC-BY 4.0 [V] | - | Source not stated on the abstract; very likely OAI [U] - **flag, avoid**; KneeCoT (banned) is in the same genre |
| Curia ViT-B/L (Raidium, arXiv 2509.06830) | DINOv2 on 200 M CT+MRI slices | yes [V] | **research-only RAIL-M** [V] | Use restriction ("research only") is a different problem than 5.a.4 covers - **avoid for a prize entry** |
| Decipher-MR, PRISM, Mr3D-VL, MARS | large MRI FMs | not shared / not stated [V] | - | unusable |
| Merlin, CT-CLIP, fVLM | CT | yes | CC-BY / CC-BY-NC-SA / check | wrong modality |

Net: no public, non-OAI, permissively licensed *knee MRI* encoder exists with classification evidence. The
realistic "different pretraining" options are **DINOv3 (SSL, web)**, **RadImageNet (supervised, medical)**, and
**MedSigLIP / MedImageInsight (image-text, medical)** - or our own in-domain SSL on the 24k training series
(Azizi et al., "Big Self-Supervised Models Advance Medical Image Classification", ICCV 2021 [K]: in-domain SSL
after ImageNet init beats ImageNet alone on dermatology/CXR).

### Q3. Knee MRI abnormality detection SOTA (2022-2026)

- **MRNet** benchmark (Bien et al., PLoS Med 2018) [V]: 1,370 exams, 3 labels; original AUC abnormal 0.937 /
  ACL 0.965 / meniscus 0.847. ELNet (MIDL 2020) meniscus 0.904 / ACL 0.960 [V]; MRPyrNet (MIDL 2021, feature
  pyramid + detail pooling, a plug-in that helped both MRNet and ELNet) [V]; SB-SSL (slice-based SSL
  transformers, 2022) [V title]. 2024-2025 journal claims of 0.97-0.99 AUC or "99.94 % accuracy" (KneeXNet,
  MV2SwimNet, Frontiers 2025) come with test-set reuse and no site split; **treat as unreliable**. Robust
  takeaways: per-plane 2-D encoders + slice pooling are the standard; meniscus is the hardest label everywhere.
- **Multi-task 9-abnormality DLS, eClinicalMedicine 2025** (Qiu et al., 13,419 patients / 14,962 exams, 6 Chinese
  hospitals) [V]: three per-plane ResNet encoders, features concatenated; **3D U-Net meniscus segmentation to
  crop a 256x256 ROI**; attention-guided object localisation (coarse-to-fine). AUC internal 0.898 -> external
  0.852 / 0.812 (primary lesions). Labels by 3 MSK radiologists from images + reports. Code
  `github.com/zqiuak/KAMRNet`, **no weights**. Takeaway: localisation-then-classify is used in the best large
  study, but external AUCs (0.81-0.85) are below ours, so it is not proof of superiority.
- Eur Radiol 2025 "Comprehensive DL-assisted multi-condition analysis of knee MRI" (23 conditions) [V title, text
  paywalled] - multi-condition 2-D approach; details not verified.
- **2D-MIL vs 3D**, Harvey et al., arXiv 2604.26807 (Apr 2026) [V]: across 3 CT + 4 MRI datasets, **simple
  mean-pooling MIL over slice embeddings matches or beats attention-MIL, 3D CNNs and 3D ViTs on 4 of 6
  moderate-size tasks, 25x faster to train**. Supports our 2.5-D choice and says a 3-D pipeline B is *not*
  expected to be stronger - only more different.
- Meniscus benchmarks MeniMV (arXiv 2512.18437) and MeniOmni (arXiv 2605.28161) [V titles]: multi-view severity
  grading; nothing transferable beyond multi-view fusion we already do.
- **What plausibly transfers to our 12 labels**: (a) ROI crops per structure (we already gained from the 90 mm
  sagittal meniscus view; the eClinicalMedicine system crops by segmentation); (b) multi-plane feature
  concatenation (we do); (c) fine-grained multi-task heads (their 9 labels include PCL/LCL/cartilage/plica/IFP/
  cyst - supervision we throw away today, see R2).

### Q4. Noisy / weak report-derived labels, beyond what we did

- **LNMBench** (arXiv 2512.09315, Pattern Recognition 2026) [V via audit]: generic LNL methods (co-teaching,
  DivideMix-type) degrade substantially under real-world noise. Classic methods [K]: co-teaching (NeurIPS 2018),
  DivideMix (ICLR 2020), **ELR - early-learning regularisation** (NeurIPS 2020; target = temporal ensemble of the
  model's own predictions, i.e. in-run self-pseudo-labelling), GCE / SCE robust losses (NeurIPS 2018 / ICCV 2019).
- **Missing labels / "not mentioned" is not negative** - the relevant literature is multi-label learning with
  *missing / single-positive* labels, not generic noise:
  - Cole et al., "Multi-Label Learning from Single Positive Labels", CVPR 2021, arXiv 2106.09708 [V]: "assume
    negative" (AN) is biased; ROLE jointly estimates the unobserved labels; expected-positive regulariser.
  - Kim et al., "Large Loss Matters in Weakly Supervised Multi-Label Classification", CVPR 2022 [K]: under AN, the
    false negatives are the **large-loss** assumed-negative cells; rejecting / correcting them online (LL-R/LL-Ct)
    gives most of the gain at zero cost. Directly maps to our silent synovitis/fracture cells.
  - Zhou et al., "Acknowledging the Unknown for Multi-label Learning with Single Positive Labels", ECCV 2022,
    arXiv 2203.16219 [V]: entropy-maximisation on unknown cells + asymmetric pseudo-labelling.
  - PU-MLC (Yuan et al., arXiv 2306.16016) [V]: discard negatives, train with positive + unlabelled, with
    re-balancing - fits labels where negation is rare (synovitis, Baker's).
  - VisualCheXbert (Jain et al., ACM CHIL 2021) [V title]: report labels systematically disagree with image
    labels; learning a report->image-label mapping beat report labels - this is what our fold-local
    calibration + teacher mixing already approximates.
- **AUC-oriented loss**: Yuan et al., "Large-scale Robust Deep AUC Maximization" (AUC-margin loss, LibAUC),
  ICCV 2021 [K] - won CheXpert; typical recipe is CE-pretrain then AUC-margin fine-tune. Gain over well-tuned
  BCE is usually small (~0.5-1 pt on CXR) [K/U]. Useful mainly as a *diversity* lever for pipeline B.
- **Label smoothing vs soft targets**: Rep-GLS (arXiv 2508.02495, audit) = report-uncertainty smoothing ≈ our
  hedged/mild mapping. Nothing new.
- **Self-training / distillation**: Noisy Student (Xie et al., CVPR 2020) [K] - the student must be *noised*
  (strong aug, dropout, stochastic depth) and ideally equal/larger than the teacher; Beyer et al., "Knowledge
  distillation: A good teacher is patient and consistent", CVPR 2022 [K] - long schedules, identical crops for
  teacher/student. Our round-2 failure (teacher = own students) matches the known confirmation-bias failure;
  the literature fix is a **different** teacher, not another round.

### Q5. Diversity for ensembling - what the evidence says

- **Gontijo-Lopes, Dauphin, Cubuk, "No One Representation to Rule Them All", ICLR 2022** [V]: models that
  diverge more in *training methodology* (objective, data, framework) make increasingly uncorrelated errors;
  two ~76.5 % ImageNet models from divergent methods ensemble to 83.4 %. Hyper-parameter / seed changes give
  the least decorrelation; **pretraining objective and pretraining data give the most**.
- **Mustafa et al., "Deep Ensembles for Low-Data Transfer Learning"**, arXiv 2010.06866 (2020) [K]: in low-data
  transfer, ensembles over *different upstream pretrained models* beat seed ensembles of the best one.
- Fort, Hu, Lakshminarayanan, "Deep Ensembles: A Loss Landscape Perspective" (2019) [K]: seeds explore different
  modes but share inductive bias; fine-tuning from one pretrained checkpoint collapses that diversity
  (our 12 C seeds at rho ~0.92+ are exactly this).
- Entropy-guided MAE + ImageNet ConvNeXt-T ensemble (arXiv 2605.21970, 2026) [V]: in-domain-MAE-pretrained and
  ImageNet-pretrained copies of the same architecture are complementary on 4 medical datasets.
- Our own history agrees: diversity came from **labels and views** (A+E+C +0.0126 pre-PL), then **shared
  pseudo-labels from one teacher collapsed it** (rho Ap-Cp 0.925, "C alone matches every blend"). The shared
  teacher is the largest correlation source we have introduced. Also our note: rho alone does not rank arms
  (C contributed more than E at higher rho) - judge by paired blend gain.
- **Ranking of diversity levers for B** (evidence + our history): (1) teacher / target source,
  (2) pretraining objective + data, (3) input representation (resolution, slice sampling, 3-D), (4) architecture
  family, (5) loss, (6) seeds/HPs (last; measured ≈ 0 for us).

---

## 2. Ranked ideas for pipeline B

Common definitions: GPU-h = A6000 hours; "eng-days" = one developer. Gains are **guesses** in paired gold-58 of
the *merged* ensemble (A-pipeline + B) vs A alone, unless marked; LB transfer ~0.5-0.8 of gold (our §4.5 note).

### R1. Independent teacher: B gets its own round-0 -> round-1 pseudo-label chain  (MUST-HAVE)
- **Mechanism**: B trains round-0 on the report-voter soft targets (v1_1_gemma3), makes *its own* 5-fold OOF
  teacher, then round-1 students on 0.5 x label + 0.5 x **B-teacher** (never the A-pipeline's OOF). Keeps the
  biggest lever we own while removing the largest shared-error source.
- **Evidence**: our PL round +0.0135 paired gold, +0.010 LB; round-2 on own students negative (confirmation
  bias); Noisy Student / Beyer 2022; Gontijo-Lopes (training-method divergence -> decorrelation).
- **Expected gain (guess)**: without R1, B's students inherit A's teacher and B is just another C-seed
  (tie). With R1: enables everything else below; alone on an identical backbone maybe +0.002-0.004.
- **Cost**: 2 rounds x 5 folds + full-data seeds ≈ 25-40 GPU-h; 0.5 eng-day (config + target builder exists).
- **T4/licence**: n/a / ours. **Independence**: changes the *targets* (teacher half) - high.

### R2. Fine-grained report supervision: 30-50 auxiliary findings from a new LLM pass  (HIGHEST-VALUE text use)
- **Mechanism**: re-run our vLLM extractor (Qwen3-32B-AWQ, or Gemma-3 for a different voter lineage) with an
  extended schema: PCL, LCL, popliteus, patellar/quadriceps tendon, per-compartment cartilage grade (medial
  femoral/tibial, lateral, patella, trochlea), marrow oedema location (MFC/LFC/MTP/LTP/patella), meniscus tear
  horn + type (radial/horizontal/complex/root/extrusion), parameniscal cyst, plica, Hoffa oedema, loose body,
  chondrocalcinosis, postoperative ACL graft, normal-study flag, severity per item. Train B with 12 main heads +
  N auxiliary heads (loss weight 0.2-0.5, masked where "absent"). Aux heads are discarded at inference.
- **Evidence**: Reality Check (IPMI 2025) - fine-grained label supervision ≥ VLP on limited noisy text;
  MedKLIP/KAD entity supervision; Merlin (codes + text > either); eClinicalMedicine 2025 9-label multi-task;
  Bergström - text helps where the concept is in the text (these auxiliary findings are text-reported). The
  under-reported targets benefit indirectly: cartilage grades and marrow-oedema location are the report's
  proxies for OA and contusion; synovitis co-occurs with effusion/Hoffa/plica mentions.
- **Expected gain (guess)**: B single model +0.000-0.005 gold vs a B without it; more importantly its
  representation diverges from A's (different target space). Risk: aux labels dilute capacity - use weight 0.3.
- **Cost**: schema + prompt 1 eng-day; LLM pass ~2-4 GPU-h per variant; gold check not possible for aux
  (use evidence-quote check + 50-report spot check, 0.5 day). Zero inference cost.
- **T4/licence**: no inference change / ours (Qwen/Gemma licences already accepted for labels).
- **Independence**: changes the *target space* and what the encoder must represent - medium-high.

### R3. Different pretraining objective + family: DINOv3 (SSL) backbone  (primary backbone candidate)
- **Mechanism**: replace ConvNeXt-V2 FCMAE (masked-autoencoding, IN-22k fine-tuned) by a DINOv3 model:
  **ViT-B/16** (different family, patch tokens) at 256-320 px, or the distilled **ConvNeXt-S DINOv3** if ViT is
  unstable (our deep-head divergence suggests ViT needs longer warmup / LLRD 0.75 / lr 2e-5..5e-5).
- **Evidence**: Gontijo-Lopes 2022 and Mustafa 2020 (upstream model diversity > seeds); DINOv2/v3 in the public
  0.943 stacks (audit); A5v result shows *backbone pretraining* is what made C strong (ConvNeXt-S on C's views
  did not match) - pretraining really changes the function here.
- **Expected gain (guess)**: single model -0.01..0 gold vs C; blend with C +0.003-0.008 gold if rho <= 0.85.
- **Cost**: 1-2 eng-days (timm model, LLRD, token-size/speed tuning); ViT-B@256 ≈ 1.5-2x C's per-image FLOPs [U]
  -> ~3 h/fold.
- **T4**: ViT-B/16@256 fp16 fine; budget 3-6 members. **Licence**: DINOv3 License (not ND; check redistribution
  clause before release) [V card]. **Independence**: pretraining objective + data + family - high.

### R4. Different input representation: full-coverage slice sequence at lower resolution
- **Mechanism**: instead of K=16/24 sampled triplets per view @384, feed **all slices** (≤ 40, single-channel
  replicated or 3-slice sliding) at **256-288 px** per view, pooled by **mean-pool MIL + a 1-layer slice
  transformer with positional (z) encoding**, plus ROI views of different geometry than A: 100 mm coronal
  (MCL-safe, measured in §4.2) and a 90 mm axial patellofemoral crop (pf_oa). No z-sampling randomness at test.
- **Evidence**: Harvey et al. 2026 (mean-pool MIL ≥ attention-MIL / 3D on moderate MRI tasks); community
  224-288 px "sweet spot" (audit); eClinicalMedicine ROI crops; Gontijo-Lopes (input pipeline divergence).
- **Expected gain (guess)**: single model ≈ C -0.005; decorrelation moderate; +0.002-0.004 in blend.
- **Cost**: 2 eng-days (data loader + inference path + parity test); cheaper per study than C (fewer px).
- **T4**: faster than C. **Licence**: n/a. **Independence**: input representation + pooling - medium.

### R5. In-domain self-supervised pretraining on our own 24k series
- **Mechanism**: continue pretraining the B backbone on the ~819k training slices (all 6 series types, incl.
  unused `ax_t1`, `cor_t1`, siblings) before supervised training: DINO-style (for ViT) or FCMAE/MAE (for
  ConvNeXt). Only train-split images - never test.
- **Evidence**: Azizi et al. ICCV 2021 (in-domain SSL after ImageNet init helps), RAD-DINO/Curia (in-domain SSL
  FMs), 2605.21970 (in-domain MAE model complements ImageNet model in ensembles).
- **Expected gain (guess)**: +0.000-0.006 single model, more diversity. Real risk: 1-2 days of GPU with no
  guarantee, and SSL recipes need tuning.
- **Cost**: 30-50 GPU-h (both GPUs ~1 day), 1.5 eng-days. **T4/licence**: no inference change / inherits init
  licence. **Independence**: pretraining data - high. **Only if R3 lands early**; not on the critical path.

### R6. Missing-label losses for under-reported findings (B's own recipe for silent cells)
- **Mechanism**: for synovitis, fracture, contusion, Baker's, OA on *round-0* only: treat "absent" as unknown
  instead of calibrated-negative-with-weight-0.1: Large-Loss rejection (Kim CVPR 2022) or entropy-max on unknown
  cells (Zhou ECCV 2022). Round-1 then uses B's own teacher (R1).
- **Evidence**: Cole 2021, Kim 2022, Zhou 2022, PU-MLC 2023; our gold shows silent cells are the loss mass.
- **Expected gain (guess)**: round-0 B teacher better on silent cells (+0.005-0.02 on synovitis gold), which
  flows into R1's students; net merged +0.001-0.004.
- **Cost**: 0.5-1 eng-day, no extra GPU. **Independence**: target handling - medium.

### R7. AUC-margin loss fine-tune (LibAUC) as the final 2 epochs
- Mechanism: BCE for 6 epochs, then AUC-margin (Yuan ICCV 2021) 2 epochs. Evidence: CheXpert #1 [K]. Gain
  (guess) ≤ +0.003 single; mostly diversity. Cost 0.5 eng-day. MIT library [K]. Independence: loss - low-medium.
  **Bundle item, not a bet.**

### R8. Text-embedding alignment auxiliary loss (the "direct text supervision" test)
- **Mechanism**: frozen multilingual text encoder (BGE-M3 or multilingual-e5-large, MIT [K]) embeds the report
  findings (or per-label evidence sentences our LLM already extracted); B's study embedding gets (a) an
  in-batch InfoNCE loss vs its own report (MedCLIP-style soft negatives from label similarity to avoid the
  normal-report false-negative problem) or (b) a cosine-regression to the report embedding. Weight 0.1-0.2.
  Frozen text tower = MedMAP's choice; required because 4.4k pairs cannot train a text tower.
- **Evidence**: weak for our setting. Bergström: helps well-reported labels only; Reality Check: labels ≥ text;
  every positive VLP result uses ≥ 7k-200k pairs. No multilingual evidence.
- **Expected gain (guess)**: 0 ± 0.004. Cost 1.5 eng-days + CPU/GPU embedding pass (< 1 GPU-h). Zero inference
  cost. Independence: training objective - medium. **Build only as an ablation flag inside B; drop if fold-0
  paired gold is not ≥ 0.**

### R9. Medical text-aligned encoder member (MedSigLIP-448 or MedImageInsight)
- Mechanism: fine-tune the vision tower (LLRD, low lr) in B's MIL at 448/384 px, 1-2 views. Evidence: trained on
  MRI slices + text [V]; no knee classification evidence. Gain (guess) -0.02..+0.005 single; high diversity.
  Cost 2 eng-days, ~8x C's FLOPs [U] -> 5-8 h/fold. T4: 1-3 members max. Licence: MedSigLIP HAI-DEF terms
  (custom) / MedImageInsight MIT. **Backup to R3 if DINOv3 underperforms by > 0.015 gold.**

### R10. Localisation-then-classify / 3-D encoders - NOT recommended in the window
- Segmentation-driven ROI (eClinicalMedicine 2025) needs a non-OAI knee segmenter we do not have; true 3-D
  (MedicalNet, SAM-Med3D, OrthoDiffusion) has no evidence of beating 2.5-D (Harvey 2026) and 3-5 eng-days of
  plumbing. LUPI text-teacher distillation (Bergström: fails when text predicts the label) and auxiliary report
  generation (no evidence) - **skip**.

---

## 3. Recommended pipeline B (12 days build + train, 2 A6000)

**B = "DINOv3-ViT, full-coverage 288 px slice sequences, fine-grained report multi-task, own teacher."**
= R1 + R2 + R3 + R4 (+ R6 on round 0; R8 as a one-fold ablation flag; R5 only if ahead of schedule).

What changes vs pipeline A (independence ledger):

| Axis | Pipeline A (C recipe) | Pipeline B |
|---|---|---|
| Pretraining | ConvNeXt-V2 FCMAE, IN-22k | DINOv3 SSL (LVD-1689M), ViT-B/16 (fallback ConvNeXt-S DINOv3) |
| Family | ConvNeXt CNN | ViT |
| Input | K-sampled triplets @384, 5 views (130 mm x4 + 90 mm sag) | all slices (≤ 40) @288, views: 130 mm sag/cor/ax + 100 mm cor + 90 mm sag + 90 mm axial PF crop |
| Pooling | gated attention MIL + concat fusion | mean-pool + 1-layer z-transformer per view, token fusion |
| Targets | 12 labels, v1_1 + A-teacher PL | 12 labels + 30-50 aux findings (new LLM pass), round-0 missing-label loss, **B-teacher** PL |
| Loss | BCE soft | BCE soft (+ AUC-margin last 2 epochs as flag) |

**Must stay shared** (for honest evaluation and a mergeable result): `data/folds/folds_v1.csv` (md5-pinned);
gold-58 excluded from every B training set and B's round-1 teacher built only from B's leak-free OOF; the base
voter soft targets / fold-local EB calibration code (`labels/targets.py`) for the 12 labels; metrics and the
paired-bootstrap gold-58 evaluator (`scripts/evaluate_runs.py`, §4.2 rule); cache v1 (B reads it and derives
288 px on the fly - no second cache); the DICOM decode / slot-routing / laterality path in the Kaggle notebook
(one preprocessing, two model heads, so the merged kernel decodes once). Decision rule for merging: paired
gold-58 of (A-final + B-final, equal weight per *pipeline*) vs A-final, pre-declared, then **one** paired LB
probe; accept at the §4.5 thresholds (≥ +0.004 gold, LB not worse).

**Schedule (guess, 12 days from 2026-10-01, new-ideas freeze 10-15 means B's design must be frozen by then;
training may continue to ~10-18):**
- D1-2: R2 schema + LLM pass on GPU 0 (runs overnight); B data loader/model (R3+R4), parity test; fold-0
  smoke of DINOv3-ViT-B vs DINOv3-ConvNeXt-S at 200 studies (decide on train loss/throughput, not smoke AUC -
  our recorded ±0.04 smoke noise).
- D3-5: round-0, 5 folds (both GPUs, ~3 h/fold -> ~8 h wall), with R2 aux + R6; fold-0 extra run with R8 flag.
  Gate G1: B-round-0 OOF gold within 0.02 of C-round-0 (0.888) and blend(C_r0, B_r0) > C_r0 paired.
- D6-8: build B-teacher from B-round-0 OOF; round-1 5 folds (for gold evaluation) + 3 full-data seeds.
  Gate G2 (by 10-12): paired gold-58 of merge vs A-final ≥ +0.004 and B single ≥ 0.905.
- D9-10: export, merged kernel (A's C x6-12 + B x3-6), runtime ≤ 6 h projected, kernel-vs-local check,
  one LB probe (before 10-15 so the merge counts as "decided").
- D11-12: extra B full-data seeds / R5 if time; buffer for the release hazard (bundle keys, notebook rebuild).
- GPU total ≈ 60-90 GPU-h (+ 4 h LLM). Kaggle: B ViT-B@288, 6 views x ≤ 40 slices ≈ 1.2-1.5x C per member
  [U]; 12 C + 6 B fits the 9 h budget with margin given main_v6's ~2.2 h projection [U].

**Expected result (guess)**: B single ~0.905-0.920 gold; rho(B, C) ~0.80-0.88; merged +0.004-0.010 paired gold,
+0.002-0.006 LB. Probability that the merge clears the §4.5 bar: ~40-50 % [U]. If G1 fails with DINOv3, swap to
R9 (MedSigLIP) only if ≥ 5 days remain; otherwise keep B's round-0 as a small-weight member only if the paired
blend is positive, and stop.

**Why not a text-aligned (CLIP) pipeline as B's core**: at 4.4k multilingual pairs and with labels that are
themselves read from the text, the 2025 evidence (Bergström; Reality Check; all positive VLP results at ≥ 7k-200k
pairs) predicts little independent signal, and none on the under-reported labels where our error is. The report
text is best used as *more, finer labels* (R2), which the same papers support.

---

## Sources (primary links)

- Bergström et al. 2025, arXiv 2510.24385: https://arxiv.org/abs/2510.24385
- Silva-Rodríguez et al., IPMI 2025, arXiv 2504.05227: https://arxiv.org/abs/2504.05227 · code https://github.com/jusiro/DLILP
- MedMAP / MedMoM-MRI3D, arXiv 2602.23652: https://arxiv.org/abs/2602.23652
- fVLM, ICLR 2025, arXiv 2501.14548: https://arxiv.org/abs/2501.14548 · https://github.com/alibaba-damo-academy/fvlm
- Decipher-MR, arXiv 2509.21249: https://arxiv.org/html/2509.21249v2
- PRISM, arXiv 2508.07165: https://arxiv.org/abs/2508.07165 · Mr3D-VL arXiv 2608.12689: https://arxiv.org/pdf/2608.12689
- Merlin, Nature 2026: https://www.nature.com/articles/s41586-026-10181-8 · https://huggingface.co/stanfordmimi/Merlin
- KAD (MIT): https://github.com/xiaoman-zhang/KAD · CXR-CLIP arXiv 2310.13292 · "Language over Labels" https://aclanthology.org/2022.aacl-srw.11/
- PadChest-GR: https://ai.nejm.org/doi/full/10.1056/AIdbp2401120
- DINOv3 ConvNeXt (timm): https://huggingface.co/timm/convnext_tiny.dinov3_lvd1689m · https://github.com/facebookresearch/dinov3
- MedSigLIP: https://huggingface.co/google/medsiglip-448 · MedImageInsight arXiv 2410.06542: https://arxiv.org/pdf/2410.06542
- MRI-CORE arXiv 2506.12186: https://arxiv.org/html/2506.12186v1 · https://github.com/mazurowski-lab/mri_foundation
- SAM-Med3D: https://github.com/openmedlab/SAM-Med3D
- OrthoDiffusion arXiv 2602.20752: https://arxiv.org/html/2602.20752 · https://github.com/lt-0123/OrthoDiffusion
- KneePreM arXiv 2609.31461 (OAI): https://arxiv.org/abs/2609.31461 · 3DReasonKnee arXiv 2510.20967: https://arxiv.org/abs/2510.20967
- Curia arXiv 2509.06830: https://arxiv.org/abs/2509.06830 · https://huggingface.co/raidium/curia
- MRNet (PLoS Med 2018): https://journals.plos.org/plosmedicine/article?id=10.1371%2Fjournal.pmed.1002699 · ELNet arXiv 2005.02706 · MRPyrNet https://proceedings.mlr.press/v143/dunnhofer21a.html
- Knee 9-abnormality DLS, eClinicalMedicine 2025: https://pmc.ncbi.nlm.nih.gov/articles/PMC12508577/ · code https://github.com/zqiuak/KAMRNet
- Eur Radiol 2025 multi-condition knee MRI: https://link.springer.com/article/10.1007/s00330-025-12052-8
- MIL vs 3D benchmark, arXiv 2604.26807: https://arxiv.org/abs/2604.26807
- Cole et al. CVPR 2021, arXiv 2106.09708: https://arxiv.org/abs/2106.09708 · Zhou et al. ECCV 2022, arXiv 2203.16219 · PU-MLC arXiv 2306.16016: https://arxiv.org/abs/2306.16016
- VisualCheXbert arXiv 2102.11467: https://arxiv.org/pdf/2102.11467
- Gontijo-Lopes et al., ICLR 2022, arXiv 2110.12899: https://arxiv.org/abs/2110.12899
- Entropy-guided MAE ensemble, arXiv 2605.21970: https://arxiv.org/abs/2605.21970
- [K] items (not re-opened today): Kim et al. CVPR 2022 (Large Loss Matters); Yuan et al. ICCV 2021 (LibAUC);
  Liu et al. NeurIPS 2020 (ELR); Li et al. ICLR 2020 (DivideMix); Han et al. NeurIPS 2018 (co-teaching);
  Xie et al. CVPR 2020 (Noisy Student); Beyer et al. CVPR 2022 (patient teacher); Azizi et al. ICCV 2021;
  Mustafa et al. arXiv 2010.06866; Fort et al. arXiv 1912.02757; MedCLIP EMNLP 2022; GLoRIA ICCV 2021;
  BioViL-T CVPR 2023; MGCA NeurIPS 2022; PRIOR ICCV 2023; MedKLIP ICCV 2023; RadImageNet Radiol AI 2022.
