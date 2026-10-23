# Audit 2026-09-25 - ANALYST: where performance is actually lost

Scope: every OOF we own (MVP, E=eff_effv2s_256, A=main_convnext_s_384, A8=..._e8, C=main_armc_convnextv2_t_384),
all scored on the **targets v1** yardstick; shipped main entry = mean(A+E+C) = `main_v3` (LB 0.933). CPU only, no
GPU/training/submissions. Every number below was produced by the scripts in `qa/audit_2026-09-25/analyst/`
(`common.py`, `s1_labels.py` ... `s6_budget.py`; tables `s*_*.md/csv` beside them; run commands in `common.py`).
Bootstraps are study-level and paired (seed 2026). Resolution limit per §4.2: nothing below 0.005 is a finding.

## 1. Where is AUC lost? (`s1_labels.md`)

Macro, A+E+C: **weak 0.9341 / strict 0.8685 / consensus 0.8900 / gold 0.9044** (singles: C 0.9254/0.8576/0.8778/0.8878,
A8 0.9240/.../0.8863, A 0.9215/.../0.8905, E 0.9080/.../0.8666). The weak-gold gap of the blend is **0.0297**.

| label | weak A+E+C [CI95] | gold A+E+C [CI95] | weak-gold | blend - best single, weak [CI] | label ceiling v1 / g3 | gold loss share |
|---|---|---|---|---|---|---|
| lat_oa | 0.873 [0.853,0.891] | **0.778** [0.563,0.953] | +0.095 | +0.003 [-0.006,+0.010] (A8) | 0.810 / 0.804 | **0.0185** |
| synovitis | 0.955 [0.922,0.980] | 0.824 [0.708,0.923] | **+0.130** | +0.007 [-0.009,+0.025] (C) | 0.633 / 0.648 | 0.0146 |
| pf_oa | 0.919 [0.909,0.929] | 0.852 [0.734,0.956] | +0.067 | +0.008 [+0.003,+0.013] (A) | 0.867 / 0.861 | 0.0123 |
| lat_men | 0.927 [0.914,0.939] | 0.860 [0.749,0.943] | +0.067 | +0.010 [+0.002,+0.018] (C) | 0.814 / 0.823 | 0.0117 |
| contusion | 0.922 [0.905,0.938] | 0.868 [0.764,0.949] | +0.054 | **-0.003** [-0.012,+0.005] (A8) | 0.737 / 0.813 | 0.0110 |
| fracture | 0.916 [0.892,0.938] | 0.917 [0.822,0.987] | -0.001 | **-0.001** [-0.016,+0.013] (A8) | 0.796 / 0.785 | 0.0069 |
| mcl | 0.889 [0.857,0.919] | 0.930 [0.848,0.988] | -0.041 | +0.013 [-0.001,+0.027] (C) | 0.948 / 0.975 | 0.0059 |
| baker | 0.972 | 0.948 | +0.024 | +0.003 (A8) | 0.905 / 0.905 | 0.0044 |
| effusion / med_oa / med_men / acl | 0.959 / 0.946 / 0.962 / 0.972 | 0.965 / 0.966 / 0.966 / 0.980 | <= 0 | +0.003..+0.007, all P>=0.99 | 0.750 / 0.871 / 0.909 / 0.964 | 0.002-0.003 each |

Label ceiling = AUC of the fold-honest calibrated soft target vs gold (the labels' own agreement with gold): macro
**0.834 (v1) / 0.848 (v1_1_gemma3)**. The blend (0.904 gold) already **exceeds its own labels by +0.07**; it is below
the ceiling only on **lat_oa (-0.032), mcl (-0.018), pf_oa (-0.015)** - the labels where the image model, not the
report, is the bottleneck. Everywhere else the model beats its labels (effusion +0.21, synovitis +0.19, contusion +0.13).

**Ranking by recoverable headroom.** (1) *Weak->gold definition gap*: lat_oa, synovitis, pf_oa, lat_men, contusion carry
**0.068 of the 0.096 gold shortfall (71 %)** while scoring 0.87-0.96 on weak labels - the loss is in what reports call
positive vs what the host's readers call positive (severity thresholds), not in fitting the reports. (2) *Image-limited*:
lat_oa (weakest on weak 0.873 AND gold 0.778), mcl (weak 0.889), pf_oa. (3) *Ensemble*: the blend is not the best
option only on contusion/fracture (A8 better by 0.003/0.001, both ties). Gold per-label CIs are 0.1-0.4 wide: per-label
gold numbers rank, they do not measure.

## 2. Slices (`s2_slices.md`, `s2b_nseries.md`)

A+E+C weak macro inside each slice vs the whole-population macro **on the same labels** (labels need >=10 pos/neg in the
slice); CI = 500 bootstrap reps in the slice. Marked-worse slices (CI excludes ref):

| slice | n | slice [CI95] | ref | delta | consensus delta | worst label |
|---|---|---|---|---|---|---|
| n_series 8+ | 416 | 0.899 [0.876,0.918] | 0.932 | **-0.034** | -0.022 | fracture -0.118 |
| site Philips Achieva dStream 3T en | 136 | 0.902 [0.864,0.937] | 0.937 | **-0.035** | -0.020 | lat_oa -0.165 |
| site Siemens Vida XA60 hr | 118 | 0.913 [0.873,0.947] | 0.944 | -0.031 | -0.018 | acl -0.093 |
| lang bg (= one Philips Ingenia site) | 217 | 0.925 [0.897,0.947] | 0.944 | -0.019 | -0.026 | pf_oa -0.091 |
| manufacturer Philips | 1,283 | 0.923 [0.910,0.935] | 0.934 | -0.012 | -0.010 | mcl -0.072 |
| cor_fs missing -> cor_t1 fallback | 157 | 0.924 [0.896,0.952] | 0.939 | -0.015 (CI covers) | -0.052 | contusion -0.134 |

**The 8+ series deficit is a site effect, not a series-selection bug**: 130 of the 416 are the Achieva dStream site; on
the 3 sites with both kinds of study, 8+ = 0.937 vs <8 = 0.936. No pipeline slice is worse: 3D in a main slot +0.006
(n=515), sag_fs missing->sag_t1 +0.001 (n=255), FOV<130 mm n=2, 1.5 T -0.004 / 3 T +0.004, sibling count no trend.
Better-than-average slices are GE (+0.021, n=889) and "side unknown" (+0.019, n=875; 74 % GE + 26 % Toshiba vs 7 % GE elsewhere - a site proxy).
Language: only bg is worse; es/tr/el are *better* (+0.011/+0.013/+0.018). Per-fold spread -0.004..+0.018 = site mix.
Worst sites cover 3-5 % of studies each: fixing one completely is worth <= ~0.002 macro.

## 3. Error analysis of A+E+C (`s3_errors.md`, `s3_gold_failures.csv`)

Confident error = a weak negative ranked above the label's median positive (cFP) or a positive below the median
negative (cFN): **cFP 318/22,633 (1.4 %), cFN 195/7,650 (2.5 %)**. Systematic patterns:
- **Label-side, not model-side**: cFP rate triples as voters disagree (all3 0.012 / llm2 0.019 / split 0.036); cFN in
  llm2 cells 0.052 vs all3 0.022. Contusion cFN is 0.105 on mild/hedged positives vs 0.021 on firm ones; fracture
  0.071 vs 0.045 - the "positives" the model rejects are the ones below the host's threshold.
- Per label the model's own weak spots: **lat_oa cFP 0.039 / cFN 0.068**, fracture 0.024/0.057, mcl 0.025/0.055,
  contusion cFN 0.053, pf_oa 0.023/0.028. ACL, baker, med_men < 0.005 cFP.
- Report language: el cFP 0.041 and bg 0.031 vs en 0.010 / es 0.009 - extractor/report style, not image.
- **Gold-58**: per-arm cell failure rate (misrank share > 0.5) MVP 0.085, E 0.070, A 0.050, A8 0.047, C 0.049, blend
  0.042. **10 of 696 cells fail in every arm**: lat_oa 4, synovitis 2, contusion 2, pf_oa 1, fracture 1 (5 missed
  positives, 5 false alarms); in **6 the weak label points the same wrong way** (label-driven), in 4 the label was right.
  Studies (last 10 digits of the UID): ...3216559395 lat_oa+ and ...3451855003 lat_oa+ (both en, label "absent"),
  ...2853419550 fracture+ (en, absent), ...8836556945 synovitis+ (es, absent), ...0594649149 pf_oa+ (es, label
  severe - image-driven miss), ...6968676828 lat_oa- (hr, label moderate), ...3432197492 lat_oa- (el, mild),
  ...8921482192 contusion- (en, affirmed), ...0568677187 contusion- (tr), ...1908452608 synovitis- (tr). No study
  fails on >= 3 labels - there is no "broken study" (decode/series) pattern on gold.
- **Laterality is not a real error source.** In weak studies with exactly one meniscus positive, rank(med)-rank(lat)
  separates med+/lat- from lat+/med- at **AUC 0.978 [0.966,0.988]** (control med_men vs acl, unrelated findings: 0.989);
  left 0.983 [0.967,0.994], right 0.967 [0.940,0.988], side unknown 0.989 - no side asymmetry. OA pair 0.878
  [0.810,0.934] with the same L/R overlap (0.923 vs 0.855) - explained by lat_oa being the weakest label, not by
  swapped sides. Gold: menisci 2/25 discordant studies swapped, OA 4/14. Flip canonicalisation (§2.3/A3) is capped
  at a few thousandths on 4 labels -> < 0.001 macro.

## 4. Ensemble anatomy (`s4_ensemble.md`, `s4_subsets.csv`)

Mean per-label Pearson: A-A8 **0.950**, A-C 0.883, A8-C 0.888, E-{A,A8,C} 0.832-0.836 (mcl lowest 0.69-0.78).
Leave-one-out from A+E+C (weak): dropping **C costs 0.0050, A 0.0042, E 0.0022**. Where each helps: C lat_men +0.012,
contusion +0.011, synovitis +0.009, mcl +0.007; E synovitis +0.010 but contusion -0.005; A contusion +0.009, lat_men
+0.008, pf_oa +0.008, synovitis -0.004. A8 on top of A+E+C: **+0.0007** (contusion/fracture +0.005, mcl -0.003) -
A8 is a near-copy of A. Diminishing returns: 2nd arm +0.0077 (A->A+E), 3rd +0.0049 (A+E->A+E+C), 4th +0.0007.

Best of all 31 equal-weight member sets = **A+A8+E+C+MVP: weak +0.0024 [+0.0008,+0.0040], consensus +0.0018
[+0.0001,+0.0036], gold +0.0034 [-0.005,+0.012]** vs A+E+C, before a ~one-half-width selection optimism. A+A8+E+C:
+0.0007 [-0.0001,+0.0015]. Both are **ties** under §4.2 (< 0.005) and cost +10 members. **A+E+C is the honest optimum.**

OOF -> public LB, 5 scored entries (weak / LB): MVP 0.8815/0.865, E 0.9080/0.913, A 0.9215/0.931, A+E 0.9292/0.929,
A+E+C 0.9341/0.933. OLS LB = -0.286 + 1.312 x weak, residual SD 0.0084, Pearson 0.967 (without MVP: slope 0.73).
Simulated public split (390 of the weak studies, 400 draws): level SD **0.0097**, paired-delta SD **0.0023** for
(A+E+C)-(A+E) and **0.0044** for (A+E)-A. **What it can tell us**: only that gaps >= ~0.02 transfer, and that
LB sits within +-0.01 of OOF-weak for v1-trained entries. **What it cannot**: rank any two of A, A+E, A+E+C - the
observed LB spread (0.004) is inside level noise, and the one inversion (A+E below A by 0.002 where OOF says +0.0077)
is ~2 sampling SD, i.e. consistent with a weak-vs-radiologist label shift that we cannot measure with 58 gold rows.

## 5. Compute and time budget (`s5_compute.md`, `s6_budget.md`)

Training GPU-busy hours (sum of logged train+eval s): A 15.2 h (12 ep, 203 img/s), C 12.2 h (10 ep, 301 img/s),
A8 10.2 h (8 ep), E 4.4 h, MVP 3.4 h; aborted/side runs (coatnet 1.8, armc_v1_1_f0 2.4, armA8_v2 2.3, sweeps/smokes 1.5);
**total 53.3 GPU-h training** (LLM labelling not in these logs). **One more arm** = 10.2 (A8-style) to 12.2 GPU-h (C-style)
= ~5-6 h wall on 2 GPUs, plus 1.2 h (A-size) to 1.9 h (C-size) of Kaggle runtime. Last-epoch policy is right: best-epoch
selection would add +0.008 (A) / +0.006 (C) / +0.004 (A8) of pure optimism.

Kaggle runtime: A6000 bench for 1,300 studies = eff_v1 9.2 min, main_v2 63.3, **main_v3 133.0 min** (GPU 5.61 s/study vs
2.53 for main_v2). Measured Kaggle/A6000 factor: MVP 2.18 (25 min hidden), main_v2 <= 1.70 (submitted 14:44 UTC,
COMPLETE <= 107 min later). **main_v3 hidden runtime is not recorded anywhere** (no poll log); projection **3.8-4.8 h**
of the 9 h. Room after main_v3 (v2 factor): 5.2 h hard limit -> 2.8 C-size / 4.5 A-size arms; 9 h/1.25 -> 1.8 C / 3.0 A;
the §2.7 rule (projected x1.25 < 6 h) -> **0.6 C / 0.9 A** - i.e. under our own rule the budget holds about one more
A-size arm. Dropping z-TTA would free ~1.7 h but its AUC value was never measured.

## 6. Efficiency track (`s6_budget.md`)

Score = -AUC/0.45 + runtime_s/32400 (Base 0.5, maxAUC 0.95 assumed): **1 min = 0.00083 AUC; 0.01 AUC = 12 min.**
eff_v1: LB 0.913, ~20 min (MVP factor x 9.2 A6000-min; not measured) -> **-1.992**; arm A alone ~82 min -> -1.918;
main_v2 <= 107 min -> -1.866; main_v3 ~225 min -> -1.656. eff_v1 is our best efficiency entry and it is **prep-bound**:
decode/resample = 7.2 of 9.2 A6000-min, GPU 4.6 min overlapped - a faster or distilled *network* can save at most ~4.3
Kaggle-min = **0.0035 AUC-equivalent**. Break-even for a replacement: 5 min needs AUC >= 0.9005, 10 min >= 0.9047,
15 min >= 0.9089, 30 min >= 0.9214, 45 min >= 0.9339. The lever is therefore *more AUC at ~the same decode cost*
(e.g. a C-type model, OOF-weak 0.9254 vs E 0.9080, at 2 folds/K16/no TTA: est. 20-26 Kaggle-min, break-even LB
0.913-0.918), not a smaller network.

## 7. The 6 biggest quantified opportunities

| # | opportunity | expected gain | evidence | confidence | clears §4.2 (>=0.010, P>=0.95)? |
|---|---|---|---|---|---|
| 1 | **Close the weak->gold definition gap on 5 labels** (lat_oa, synovitis, pf_oa, lat_men, contusion): targets that encode the host severity thresholds (stage-9 finding d), not more voters | 0.068 of the 0.096 gold shortfall lives here; realistic +0.005..0.015 gold/LB, *unmeasurable offline* | weak-gold gaps +0.054..+0.130; 6/10 all-arm gold failures are label-driven; contusion cFN 5x on mild/hedged | low-medium | **Cannot be shown**: gold +-0.027, LB +-0.01; consensus/weak are blind to it by construction |
| 2 | **Efficiency entry: C-type model at 2 folds, K16, no TTA** in place of E | est. 20-26 min, break-even LB 0.913-0.918; +0.005..0.010 AUC-equiv if it lands at LB ~0.925 | C OOF-weak +0.017 over E; break-even at 30 min is 0.9214; eff is prep-bound so its runtime floor barely moves | medium | Different track; OOF gap C-E 0.017 P=1.00 clears it for AUC, runtime must be measured on Kaggle |
| 3 | **Image-limited lateral compartment**: lat_oa (weak 0.873, gold 0.778, below its label ceiling by 0.032), mcl, pf_oa - per-label view attention (arm N) / sag_t1 evidence | lat_oa to its best single-view aux AUC (0.896, arm C fold-0 sag_t1 head, stage 11) = +0.02 on 1 label = **~+0.002 macro**; all three labels +0.02 = ~+0.005 | s1 ceiling table; cFP/cFN highest on lat_oa/mcl/fracture | medium | No (<= 0.005) unless it also lifts other labels |
| 4 | **One more diverse arm** on v1_1_gemma3 targets | +0.002..0.004 weak (3rd arm gave +0.0049, a 4th A8 +0.0007; must have rho <= ~0.88 with A/C) | LOO + correlation tables | medium | No; costs 10-12 GPU-h and 1.2-1.9 Kaggle-h, and only ~1 A-size arm fits the §2.7 rule |
| 5 | **Measure main_v3's real runtime and the value of z-TTA** (read Kaggle UI; one GPU OOF pass with/without z-offset) | enabler: up to ~1.7 h Kaggle freed = room for 1-2 arms; runtime margin currently a 3.8-4.8 h *projection* | s6: no hidden-run timing recorded for main_v3 | high (that it is unknown) | n/a - a measurement, zero risk |
| 6 | **Re-blend existing arms** (A+A8+E+C+MVP) | +0.0024 [+0.0008,+0.0040] weak, +0.0018 consensus, minus selection optimism | 31-set sweep | high (that it is small) | **No - a tie**; +10 members; do not ship |

Not opportunities (measured): laterality flip/canonicalisation (< 0.001 macro; sides separated at AUC 0.978 with no L/R
asymmetry); series-count / 3D / fallback slices (site effects or no effect); single-site fixes (<= 0.002 each);
per-label blend weights (forbidden by §2.6 and worth <= 0.003 on contusion/fracture). **Bottom line: nothing we can
build and validate offline clears 0.010 any more; the ensemble is saturated (4th arm +0.0007) and 71 % of the
gold-visible loss sits in the report-vs-reader definition gap on five labels.**
