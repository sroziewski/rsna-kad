# Stage 9 - label refinement from the image models' OOF (developer J, 2026-09-21)

Two separate questions, answered separately:

1. **Confident-learning denoising** - are there weak-label cells the image models contradict
   strongly enough to be worth down-weighting or dropping? **Answer: no. Rejected on evidence.**
2. **Pseudo-labels for the under-reported keys** - can the models fill in the cells the reports
   are silent about? **Answer: the honest experiment is running; targets v2 is not shipped yet.**

Inputs: `runs/main_convnext_s_384/oof.parquet` (arm A, OOF-weak 0.9215), `runs/eff_effv2s_256/oof.parquet`
(efficiency, 0.9080), `data/targets/v1/fold{k}.parquet`, `data/labels/llm_v1.parquet`,
`data/folds/folds_v1.csv`. Nothing under `data/targets/v1` or `data/labels/llm_v1*` was modified.

Reproduce: `python qa/stage9/denoise_scan.py`, `qa/stage9/denoise_sweep.py`, `qa/stage9/leak_check.py`
(outputs committed next to them as `*.txt`; the 52 contradicted cells as `contradictions.csv`).

---

## 1. The leakage trap, and why the obvious experiment is worthless

The tempting move is: take `runs/main_convnext_s_384/oof.parquet`, build pseudo-labels from it,
retrain all five folds, read the new OOF. That number would be meaningless.

The OOF prediction for a study `s` in fold *j* comes from model `M_j`, trained on **every fold
except j** - which includes fold *k*. Build a pseudo-label for `s` from `M_j`, put `s` into the
training set of the new fold-*k* model, and fold-*k* labels have now travelled into fold-*k*'s
training data through `M_j`'s weights. Fold *k*'s validation is then optimistic, every fold is
optimistic for the same reason, and the assembled OOF rises while the LB does not.

It is worth being precise about why option (a) in the brief - "fold-honest pseudo-labels from
the 5 existing folds" - **cannot be rescued by selecting a subset of (s, k) pairs**. A pseudo-label
for `s` used by the fold-*k* model must come from a model that saw neither `s` nor fold *k*, i.e.
a model excluding *both* fold(*s*) and fold *k*. Each of `M_0..M_4` excludes exactly **one** fold.
So for any *k*, the set of usable pairs is empty - not small, empty. Honest fold-wise pseudo-labels
over all five folds need ten leave-two-out models (~20 GPU-h at the e8 cost), not five.

Option (b), nested CV over all five outer folds, is the same cost with more bookkeeping.

**Chosen: option (c)** - one outer fold, with a pseudo-label source built from scratch so that it
is provably blind to that fold.

## 2. The design

Held-out fold: **fold 0**. Two half-data source models supply every pseudo-label:

| source | trained on | labels | config |
|---|---|---|---|
| `pseudo_srcA` | folds {1, 2} (1,739 studies) | folds {3, 4} (1,765 studies) | `configs/stage9_pseudo_srcA.yaml` |
| `pseudo_srcB` | folds {3, 4} (1,741 studies) | folds {1, 2} (1,762 studies) | `configs/stage9_pseudo_srcB.yaml` |

Then `data/targets/v2/fold0.parquet` is built and `configs/main_armA8_v2.yaml` trains **fold 0
only**, byte-identical to `configs/main_convnext_s_384_e8.yaml` except `targets_version: v2`.

### Which model produced each pseudo-label

Every pseudo-labelled cell of study `s` carries the prediction of exactly one source:
`srcB` if `s` is in fold 1 or 2, `srcA` if `s` is in fold 3 or 4. No study is labelled twice
(`make_pseudo_targets.py build` aborts on a duplicate), and no fold-0 study is labelled at all.

### Proof that no fold-0 information enters fold-0 training

1. **Neither source ever sees a fold-0 study, in training or in validation.** The mechanism is a
   remapped folds CSV plus a restricted targets table: `train.py:split_studies` computes
   `train = {fold != 0} and {has a targets row} and {cached}` and `val = {fold == 0}`. The remapped
   CSV (`qa/stage9/folds_src{A,B}.csv`) puts the folds a source must *predict* into pseudo-fold 0
   and everything else - including the original fold 0 - into pseudo-fold 1; the source's targets
   table (`qa/stage9/targets/src{A,B}/fold0.parquet`) contains only the rows it may train on.
   Fold-0 studies are therefore in neither set: they are not in pseudo-fold 0, so not validated,
   and they have no targets row, so not trained. **Verified mechanically** by running
   `split_studies` itself (`qa/stage9/leak_check.txt`): srcA train = {1: 868, 2: 871},
   val = {3: 882, 4: 883}; srcB train = {3: 870, 4: 871}, val = {1: 879, 2: 883}; fold-0 studies
   in train **0**, in val **0**, train-and-val overlap **0**, gold in train **0**.
2. **Neither source labels a study it trained on.** Train and label folds are disjoint by
   construction, and (1) shows the overlap is 0.
3. **No fold-0 gold reaches a source through the target calibration.** Both source tables are a
   row subset of `data/targets/v1/fold0.parquet`, which section 2.5 calibrates on the gold rows of
   folds != 0. The fold-0 gold rows are the only gold the experiment's evaluation uses and they are
   absent from every calibration in the chain.
4. **The experiment model's training set excludes fold 0** - the ordinary `split_studies` rule for
   `--fold 0`, unchanged.
5. **No model selection touches fold 0.** All three runs use `select_epoch: last` with a fixed
   epoch count; nothing is early-stopped, and the sources' per-epoch validation numbers (on folds
   1-4) are recorded but never acted on.
6. **The pseudo-label hyper-parameters are fixed a priori, not tuned.** The 0.5-real/0.5-pseudo mix
   is verbatim from research.md section 3 (RSNA-2024 lumbar 2nd). The key set is
   `constants.UNDER_REPORTED_KEYS`, fixed at stage 2 before any image model existed. The cells
   touched are exactly the weight-0.1 cells v1 already created. The new weight, 0.5, is v1's own
   `absent_well_reported`. Nothing in this list was read off fold 0.
7. **Fold-0 rows of `data/targets/v2/fold0.parquet` are bit-identical to v1** - asserted by
   `make_pseudo_targets.py build`, which refuses to write the table otherwise. This also makes the
   comparison exact: the OOF-weak/strict/gold numbers the v2 run reports are still the **v1**
   metric, so `main_armA8_v2/fold0` and `main_convnext_s_384_e8/fold0` are paired cell for cell.

### What this design does *not* license

The sources are trained on **half** the data, so they are weaker than the real 4/5-fold models and
the measured effect is a **lower bound** on what a full pseudo-label round would give. A positive
fold-0 result therefore justifies paying for ten leave-two-out sources - it does **not** justify
shipping a five-fold v2 built from the existing (leaky) OOF. That shortcut is the exact failure
mode this section exists to prevent.

## 3. Pseudo-label rule (`src/rsna_kad/labels/pseudo.py`)

Applied only to a cell that is *silent*: one of the 7 under-reported keys whose v1 weight is 0.1,
i.e. the LLM found no mention at all. Those cells are the entire problem - synovitis is affirmed in
12.5 % of reports and positive in 46.6 % of gold studies - and they currently carry a calibrated
base rate at a weight chosen to keep the loss defined rather than to teach anything.

```
target' = 0.5 * target_v1 + 0.5 * p_source        weight' = 0.5
```

A cell outside the key set, a cell with a real report state (affirmed/hedged/negated), a row
overridden for parse failure or postoperative status, and any study without a source prediction all
keep their v1 value bit-for-bit. `tests/test_pseudo.py` pins all of that (10 tests); the full suite
is **231 passed**.

Headroom on those cells today (OOF = mean of the two arms; `qa/stage9/denoise_scan.txt`):

| key | silent cells | v1 target | OOF mean | p50 | p90 | frac > 0.5 |
|---|---|---|---|---|---|---|
| synovitis | 3,700 | 0.290 | 0.370 | 0.347 | 0.530 | 13.2 % |
| baker | 2,144 | 0.043 | 0.096 | 0.055 | 0.213 | 2.1 % |
| fracture | 2,001 | 0.139 | 0.144 | 0.118 | 0.243 | 1.7 % |
| lat_oa | 1,582 | 0.085 | 0.157 | 0.116 | 0.337 | 3.2 % |
| contusion | 1,413 | 0.072 | 0.186 | 0.148 | 0.359 | 3.4 % |
| med_oa | 1,223 | 0.096 | 0.172 | 0.113 | 0.405 | 5.5 % |
| pf_oa | 985 | 0.110 | 0.340 | 0.263 | 0.724 | 23.9 % |

13,048 cells, 25 % of the table. `pf_oa` and `synovitis` carry almost all of the disagreement;
`baker` and `fracture` will barely move, which is the honest expectation - the rule will help two
keys, not seven.

## 4. Confident-learning denoising - measured, then rejected

Cells where the mean OOF contradicts a high-weight v1 label, over 52,188 (study, label) cells:

| band (target >= hi and OOF <= lo, or target <= lo and OOF >= hi; weight >= 0.5) | label-positive | label-negative | total | share |
|---|---|---|---|---|
| **0.90 / 0.10** | 45 | 6 | **51** | 0.098 % |
| 0.85 / 0.15 | 212 | 15 | 227 | 0.435 % |
| 0.80 / 0.20 | 367 | 39 | 406 | 0.778 % |
| 0.75 / 0.25 | 598 | 88 | 686 | 1.314 % |
| 0.70 / 0.30 | 858 | 156 | 1,014 | 1.943 % |

Per key at the 0.9/0.1 band (no weight filter, 52 cells): med_men 17, pf_oa 12, acl 9, baker 5,
fracture 5, effusion 3, lat_men 1; **zero** for mcl, med_oa, lat_oa, synovitis, contusion. Both arms
agree on the contradiction in 36 of the 52. By LLM state: 45 `affirmed` contradicted downward,
6 `negated` and 1 `absent` contradicted upward.

Note the asymmetry: several keys have **no** cell at target >= 0.9 at all (mcl, lat_men, med_oa,
lat_oa, contusion), because fold-local calibration pulled their `affirmed` value below 0.9. So the
0.9 threshold is not comparable across keys, which is a second reason not to build a rule on it.

The architecture's stage-9 line anticipated relabelling "the 2 % most contradicted OOF cells". The
data says the 2 % band is at 0.70/0.30 - a band where the model is merely *uncertain*, not
contradicting. At the band that actually means "the model is confident the label is wrong", there
are **52 cells out of 52,188 (0.10 %)**.

### The 10 hand checks (full text in `qa/stage9/hand_checks.txt`)

The 10 most extreme contradictions, one per affected key then by gap size, read against the report:

| # | key | lang | LLM state | target | OOF | verdict against the report text |
|---|---|---|---|---|---|---|
| 1 | acl | fr | affirmed/moderate | 0.941 | 0.008 | label **correct** ("dechirure partielle chronique de grade bas a modere"); a low-grade partial tear is probably below the host's >50 %-of-fibres threshold |
| 2 | baker | hr | affirmed/moderate | 0.957 | 0.026 | label **correct** ("Poplitealna cista velicine 28x7x6mm"); a 7x6 mm cyst is tiny |
| 3 | med_men | en | negated/none | 0.012 | 0.913 | label **correct** ("Medial meniscus normal") - but the report is a 3-line typo-ridden template; the image disagrees loudly |
| 4 | pf_oa | de | affirmed/moderate | 0.966 | 0.069 | label **correct** ("fokal tiefgreifender Knorpeldefekt"), and the same sentence says "kein vollschichtiger Knorpeldefekt, kein subchondrales Oedem" - borderline by extent |
| 5 | fracture | bg | affirmed/unspec. | 0.954 | 0.079 | label **correct and unambiguous** (patellar lower-pole fracture + marrow oedema); the **model** is wrong |
| 6 | effusion | el | affirmed/moderate | 0.902 | 0.051 | label **correct** (moderate quantity of intra-articular fluid); moderate is above the host threshold; the model is probably wrong |
| 7 | lat_men | en | negated/none | 0.089 | 0.924 | label **correct** ("Lateral meniscus normal") while the *medial* meniscus has a bucket-handle tear - this is the known medial/lateral confusion (section 2.3), a model failure |
| 8 | acl | hr | affirmed/moderate | 0.941 | 0.017 | **LLM error**: "Blaze do umjerene promjene prednjeg krizanog ligamenta" = mild-to-moderate *changes*, not a tear |
| 9 | pf_oa | en | affirmed/severe | 0.964 | 0.070 | label **correct** ("subcentimeter focus of grade 3/4 cartilage loss") - severe in grade, negligible in extent |
| 10 | med_men | en | affirmed/unspec. | 0.909 | 0.015 | label **correct and unambiguous** ("the medial meniscus is torn"); the **model** is wrong |

Tally: **1 of 10** is a genuine label error against the report text (#8). **4 of 10** are labels
that are correct against the text but describe a finding below the host's severity/extent threshold
(#1, #2, #4, #9). **5 of 10** are correct labels the image model simply gets wrong, twice for a
reason we already know about (#7, medial/lateral confusion).

### Decision: do not down-weight, do not drop

- **Size.** 52 cells is 0.10 % of the table. Even perfect correction cannot move a 4,349-row macro
  AUC by the +0.003 the stage-9 gate requires. Widening the band to reach 2 % means calling
  "contradiction" what is really model uncertainty, and the hand checks show the *purest* end of
  the band is already half model error - the mix only gets worse as the band widens.
- **Direction.** Half the contradictions are the model being wrong about a label that is plainly
  right. Down-weighting those removes correct supervision on exactly the hard cases - patellar
  fractures, bucket-handle tears - the model most needs.
- **Precedent.** The LLM A/B study (architecture section 4.1) found the mirror image of this: an
  "error" against the report text was a *regression* against the image-derived gold, and the fix had
  to be reverted. Here the reverse holds - an apparent error against the image is mostly a correct
  label the image model has not learned yet. In both directions the lesson is the same: a
  disagreement between two noisy views is not evidence about which one is wrong.
- **What the 4 threshold cases actually say** is that the `affirmed`/severity mapping is too generous
  for low-grade partial ACL tears and small-extent PF cartilage defects. That is a *calibration*
  change to `targets.base_mild` / `MILD_PENALTY_KEYS`, not denoising, it belongs to targets v1.1,
  and v1.1 is owned by another chain. Logged below as a follow-up, not acted on here.

**No cells were down-weighted or dropped. The denoising half of stage 9 is closed as rejected.**
This also returns the ~10 GPU-h that a denoising retrain would have cost.

## 5. Status and how to finish

`data/targets/v2/` **does not exist yet** and will only be written by the queued chain, which then
aborts unless the fold-0 rows come out bit-identical to v1. Whether it survives depends on the
comparison below, which had not finished when this report was written.

Queued as screen `stage9` (`logs/stage9_20260921-170811.log`), waiting for GPU 0 to drain behind the
armA8 chain, then running `qa/stage9/stage9_chain.sh`:

```
python scripts/make_pseudo_targets.py sources                                  # done, see above
python scripts/train.py --config configs/stage9_pseudo_srcA.yaml --fold 0      # ~2.3 h
python scripts/train.py --config configs/stage9_pseudo_srcB.yaml --fold 0      # ~2.3 h
python scripts/make_pseudo_targets.py build                                    # data/targets/v2/fold0.parquet
python scripts/train.py --config configs/main_armA8_v2.yaml --fold 0           # ~2.0 h
python scripts/compare_runs.py --runs runs/main_convnext_s_384_e8 runs/main_armA8_v2 \
       --fold 0 --targets v1 --out qa/stage9/compare_v2.md
```

**Acceptance (fixed before the run, per section 4 stage 9):** keep targets v2 only if fold-0
OOF-weak improves by >= +0.003 over `main_convnext_s_384_e8/fold0` and fold-0 gold does not drop by
> 0.02. A single fold of 880 studies is a noisy read, so anything in (0, +0.003) is reported as
"no effect", not as a small win. If it clears the gate, the *next* step is ten leave-two-out sources
for an honest five-fold v2 - not a five-fold v2 from the existing OOF.

## Follow-ups for other owners

- **targets v1.1** (BUG-1/BUG-2 chain): the `affirmed` mapping is too generous where the report
  states a finding that is real but below the host threshold - low-grade partial ACL tears and
  sub-centimetre / non-full-thickness PF cartilage defects. Consider extending the
  `MILD_PENALTY_KEYS` handling with an extent qualifier, or lowering the `affirmed+mild` base for
  `acl`. 4 of our 10 hand checks are this pattern.
- **section 2.3 laterality**: hand check #7 is a clean medial-vs-lateral meniscus swap by the model
  on a report that names both explicitly. `contradictions.csv` filtered to `med_men`/`lat_men` is a
  ready-made evaluation set for the flip-canonicalisation experiment.
