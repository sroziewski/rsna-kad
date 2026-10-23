# Offline arbiter for model selection (2026-09-21)

Why: architecture §4.1 "OOF-vs-LB calibration". Four scored submissions (MVP 0.8815->LB 0.865, eff
0.9080->0.913, arm A 0.9215->0.931, blend 0.9292->0.929) show a ~390-study public LB that cannot resolve
anything below ~0.02, so every remaining decision rests on an offline metric - and the primary one,
`oof_weak_auc`, scores predictions against LLM-derived targets. This measures what that costs, tests six
candidate arbiters and states what any of them can resolve. Reproduce with `qa/arbiter/divergence.py` and
`scripts/evaluate_runs.py` (commands in their docstrings; outputs `qa/arbiter/{divergence,arbiters}.md`).

## 1. How big is the flaw?

52,884 cells: 77.1 % have three-voter agreement (V1 == V2 == rules), 14.7 % LLM-only, 8.1 % LLM split. On
the 696 gold cells the weak claim (state->score binarised at 0.5) is right 81.2 % of the time on `all3`
cells, 74.5 % on `llm2`, 73.6 % on `split`: voter agreement *does* track label quality, but only by ~7
points, and even unanimous cells are wrong 19 % of the time. The dominant error is **not** extraction
noise - it is the report-vs-image gap `qa/llm_ab/report.md` already identified.

**Do the models reproduce the extractor's mistakes?** On the 141 gold cells where the weak claim
disagrees with gold, AUC of each prediction *against the weak claim* (0.5 = ignores it, >0.5 = reproduces
it and OOF-weak pays for that): arm A **0.522 [0.310, 0.639]**, eff 0.491 [0.334, 0.640], MVP 0.475
[0.295, 0.652], blend **0.527 [0.324, 0.663]**. The blend scores highest - the direction the LB hints at
- but every CI covers 0.5 by a wide margin. **No measurable error reproduction; 58 gold studies also
cannot exclude an effect of +-0.15 AUC on those cells.**

**Does blending gain on the untrustworthy cells?** OOF-weak per cell type, delta = blend - arm A:

| cell type | kept cells | arm A | blend | delta [CI95] |
|---|---|---|---|---|
| all | 30,283 | 0.9215 | 0.9292 | +0.0077 [+0.0051, +0.0101] |
| all3 (clean) | 24,719 | 0.9281 | 0.9369 | **+0.0088 [+0.0055, +0.0122]** |
| llm2 | 4,345 | 0.8808 | 0.8862 | +0.0054 [-0.0030, +0.0139] |
| split (dirtiest) | 1,219 | 0.8443 | 0.8804 | +0.0361 [-0.0167, +0.0731] |

(Evidence-flagged cells: 290, +0.0102 [-0.0175, +0.0496].) The hypothesis "blending helps most where
labels are least trustworthy" has the right point estimate on `split` (+0.036) but a CI four times its
own width; the only *demonstrable* gain is on the cleanest 24,719 cells. **The blend's OOF gain is not an
artefact of dirty labels.** OOF-weak also already discards the worst cells: of 13,333 absent cells on
under-reported keys the weight gate (w 0.1 < 0.5) keeps **5**.

## 2. Candidate arbiters

An arbiter is a set of (study, label) cells plus a binary label; candidates are scored on exactly those
cells, so arbiters differ only in what they trust. "balanced" = AUC of the arbiter's *own* label against
gold on the gold cells it keeps (that label's balanced accuracy), i.e. how right the arbiter is.

| arbiter | cells | gold cells | label == gold | balanced | CI95 width | median paired half-width |
|---|---|---|---|---|---|---|
| **weak** (current primary) | 30,283 | 400 | 0.882 | **0.850** | 0.0107 | 0.0056 |
| weak_ev (weak & evidence_ok & no mention_conflict) | 29,993 | 395 | 0.886 | **0.853** | 0.0108 | 0.0056 |
| strict (affirmed vs negated) | 36,848 | 499 | 0.780 | 0.776 | 0.0111 | 0.0048 |
| consensus (3 voters agree, affirmed vs negated) | 28,835 | 405 | 0.798 | 0.781 | 0.0170 | 0.0061 |
| consensus+absent (unanimous silence = negative, well-reported keys) | 30,480 | 424 | 0.804 | 0.787 | 0.0169 | 0.0060 |
| consensus_ev (consensus & evidence) | 28,124 | 398 | 0.796 | 0.781 | 0.0169 | 0.0061 |
| gold58 | 696 | 696 | 1.000 | 1.000 | 0.0559 | 0.0267 |

**The consensus-subset idea is refuted by its own evidence.** It is large (28,835 cells, 12/12 labels)
and cleaner than the raw report text, but its labels are *less* faithful to the image-derived gold than
the weak targets (0.781 vs 0.850 balanced) and it is noisier (CI 0.0170 vs 0.0107). The reason is
`targets.py`: the weak label is the LLM state after fold-local empirical-Bayes calibration **on
training-fold gold rows** plus a weight gate that discards under-reported absents, voter disagreements
and hallucinated-evidence cells - exactly the gold-derived information consensus throws away. Its
per-label coverage is uneven too: synovitis 486 cells with **20 negatives**, fracture 182 positives.
Composite weights are derived, not fitted: w ~ (balanced fidelity - 0.5)/half_width^2 -> weak 0.58 /
consensus 0.38 / gold58 0.04; gold-58's +-0.027 makes it a rounding error in a scalar, hence a veto.

## 3. Ranking, and what can be resolved

| candidate | weak | strict | consensus | gold58 | composite | public LB |
|---|---|---|---|---|---|---|
| mean(main+eff) | 0.9292 | 0.8614 | 0.8832 | 0.8964 | 0.9104 | 0.929 |
| main_convnext_s_384 | 0.9215 | 0.8524 | 0.8729 | 0.8905 | 0.9018 | **0.931** |
| eff_effv2s_256 | 0.9080 | 0.8388 | 0.8615 | 0.8666 | 0.8887 | 0.913 |
| mvp_effv2s_256 | 0.8815 | 0.8095 | 0.8405 | 0.8466 | 0.8645 | 0.865 |

**All seven arbiters produce the same ordering** and all put the blend above arm A; the LB puts arm A
first by 0.002. No offline arbiter reproduces that inversion and none can - 0.002 is a fifth of the LB's
noise. The three large gaps (blend/eff 0.021, eff/mvp 0.027, arm A/mvp 0.040) are called correctly by
every arbiter, the only part of the LB signal worth agreeing with.

Paired-bootstrap 95 % CIs of a delta (2,000 study-level replicates, rows shared across candidates):

| arbiter | narrowest pair | median | widest | arm A vs blend |
|---|---|---|---|---|
| weak | +-0.0026 | +-0.0056 | +-0.0066 | -0.0076 [-0.0103, -0.0051] P=0.00 |
| strict | +-0.0024 | +-0.0048 | +-0.0058 | -0.0090 [-0.0114, -0.0066] P=0.00 |
| consensus | +-0.0033 | +-0.0061 | +-0.0074 | -0.0103 [-0.0137, -0.0072] P=0.00 |
| gold58 | +-0.0134 | +-0.0267 | +-0.0402 | -0.0062 [-0.0193, +0.0071] P=0.19 |

That is only the **sampling** term. A second, irreducible term is how far the same delta moves when the
label definition changes: arm A vs eff spans +0.0114..+0.0235 across arbiters (0.012), eff vs mvp
+0.0199..+0.0293 (0.009), arm A vs blend -0.0062..-0.0103 (0.004). The third, offline-to-LB transfer, is
unmeasurable with 4 points - the residuals of the three v1 entries (+0.005, +0.0095, -0.0002) all sit
inside the LB's own +-0.01. **Honest limit: no offline arbiter here resolves 0.005.** A delta of 0.005 at
"P = 1.000" on OOF-weak means only "this model ranks higher on these 30,283 LLM-derived cells"; the
bootstrap measures which studies were sampled, never whether the labels were right. **Spend the remaining
month on fewer, bigger bets - a change worth < ~0.01 offline cannot be validated by anything we own.**

## 4. Recommendation

**Primary arbiter: `oof_weak_auc` - unchanged.** Most gold-faithful labels (0.850 balanced), tightest CI,
12/12 labels covered, already in every `runs/*/metrics.json`; nothing measured here beats it.
**Mandatory companion: `consensus`, and *primary* whenever the runs compared were trained on different
target versions.** `oof_weak_auc` is defined *by* the target table, so when stage 9 / targets v1.1 change
the targets it changes its own yardstick and cross-version numbers are not comparable (why
`mvp_effv2s_256` reads 0.8463 in its metrics.json and 0.8815 rescored on v1); `consensus` is built from
frozen raw voter states, survives a target change, and `evaluate_runs.py` prints it for free.
**`gold58` is a veto, not a score**: at +-0.027 it can never select a model, only catch a collapse.

### How to decide

1. `scripts/evaluate_runs.py --runs <old> <new> --n-boot 2000`; read the paired delta on **weak** (same
   targets) or **consensus** (different targets).
2. **delta >= +0.010, P >= 0.95, gold-58 not down more than 0.02** -> accept.
3. **+0.005 <= delta < 0.010** -> accept only if **weak, strict and consensus agree in sign** and it
   costs no inference budget; otherwise a tie - take the cheaper side.
4. **delta < 0.005** -> **a tie, whatever P says**: decide on cost, robustness and diversity, and spend
   neither a submission nor GPU hours on it.
5. Gold-58 down > 0.02 with a weak gain -> **veto and investigate** (fitting the extractor, not the image).
6. The public LB **never** overrides an offline decision unless |delta_LB| > 0.02; the 0.002 arm-A/blend
   inversion is noise and the blend stays the main entry, as §4.1 concluded.
7. When the target table changes, re-run both scripts before comparing across it and re-derive the
   composite weights (they depend on measured fidelity and noise).

Implemented, additively: `metrics.Subset` + `weak_subset`/`strict_subset`/`consensus_subset`/
`gold_subset`, `voter_codes`, `evidence_mask`, `subset_auc`, `bootstrap_subset_auc`, `ci95`,
`paired_delta`, `weighted_arbiter`, vectorised `_rank_auc` (pinned against `auc`); `macro_auc`,
`binarise`, `auc`, `bootstrap_macro_auc`, `gold_report` untouched - the three recorded metrics recompute
exactly for `main_convnext_s_384` and `eff_effv2s_256`. Plus `scripts/evaluate_runs.py`,
`qa/arbiter/divergence.py`, and 13 new tests in `tests/test_metrics.py` (21 there, 247 in the suite).
