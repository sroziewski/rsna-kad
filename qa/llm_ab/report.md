# qa/llm_ab - is a higher-precision or better-instructed LLM worth a relabelling pass?

Developer G, 2026-09-21. Server `gpu-server2`, `/raid/sroziewski/sandbox/rsna_kad`, `.venv-llm` (vLLM 0.29 / torch
2.13), every run through `scripts/run_screen.sh`. **Answer: no. Keep `data/labels/llm_v1*` and `data/targets/v1/`
unchanged.** No `llm_v2`, no `targets/v2`.

## 1. Design

Stage 5 built `llm_v1` with Qwen3-32B **AWQ int4**; the stage-5 audit (`qa/stage5/report.md` §5) found 8 % of cells
wrong in four patterns. The quantisation or the prompt could be to blame, so this is a 2x2 of {int4, W8A16, bf16} x
{base, rewritten prompt} plus a third prompt iteration, all on one fixed subset: `qa/llm_ab/eval_ids.txt`
(deterministic, `make_eval_subset.py`) = **481 studies** = 58 gold + 400 non-gold sampled seed 0 + the 25 reports the
stage-5 tester audited by hand (23 new), so the audited patterns are counted on exactly the tester's reports. 5,772
cells per table; en 194, es 82, tr 47, hr 44, el 38, bg 29, de 24, nl 13, fr 10.

**Prompt rule sets** (`labels/prompts.py`, chosen by `LLMConfig.prompt_rules`; `base` is byte-identical to stage 5 and
`tests/test_labels_prompts_strict.py` pins it by md5): `base` = stage 5; `strict` ("V3") = same schema rewritten
against the four audited failures (STEP 1 name the structure first, STEP 2 `absent` is the default and `negated` needs
words in the report, STEP 3 group/plural statements spelled out per group incl. the scope of "no internal
derangement", STEP 4 sub-threshold findings are `affirmed`+`mild`, plus character-for-character quoting); `strict2` =
`strict` with STEP 3 given precedence over STEP 1 (see §3).

**Meniscal degeneration** (task item 2.ii) stays **`affirmed` + severity `mild`**: `state_score` already subtracts
0.35 for mild on the meniscus keys and the `affirmed_mild` target base there is 0.25 - below `hedged` (0.50) and below
the calibrated `absent`. `hedged` would *raise* the target of a finding the annotators call negative; `absent` would
discard it entirely. The mapping was never the problem, only the state assignment.

**Error detectors** (`compare.py`, proxies anchored on the multilingual term lists of `labels/rules.py`, so one
definition covers all 9 languages): `neg_unment` = `negated` though no term of that label occurs in the report;
`abs_vs_neg` = `absent` though the rule voter reads an explicit negation; `subthr_neg` = `negated` next to a
sub-threshold cue; `misattr` = verbatim quote naming another label's structure but not this one; `quote_miss`/`fabr` =
quote fuzzy ratio < 0.8 / < 0.5. On the tester's 25 reports they give 11/6/4/7 for `llm_v1` where the tester counted
14/4/6/6 by hand. The first three use the rule voter as reference, so they are only meaningful *between* LLM tables.

## 2. Results (481 studies, V1 variant, temperature 0, identical few-shots)

| table | model / serving | rules | gold-58 AUC [CI95] | parse% | verbatim% | rep/min | cell agr% | neg_unment | abs_vs_neg | subthr_neg | misattr | quote_miss | audit-25 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **awq_base** (= `llm_v1_raw`) | AWQ int4 marlin, 1 GPU | base | **0.850 [0.820,0.880]** | 100.0 | 96.6 | **38.4** | - | 127 | 260 | 178 | 105 | 141 | 11/6/4/7 |
| bf16_base | bf16, TP=2, 2 GPUs | base | 0.847 [0.814,0.878] | 100.0 | 96.7 | 33.5 | 96.2 | 135 | 218 | 193 | 100 | 139 | 11/6/7/7 |
| fp8_base | FP8 ckpt -> W8A16 marlin, 1 GPU | base | 0.845 [0.813,0.876] | 100.0 | 96.8 | 26.5 | 96.1 | 126 | 248 | 188 | 88 | 130 | 11/7/5/6 |
| awq_strict | AWQ int4 | strict | 0.848 [0.819,0.877] | 100.0 | 97.7 | 31.4 | 94.5 | 104 | 368 | 189 | 88 | 90 | 7/6/2/2 |
| awq_strict2 | AWQ int4 | strict2 | 0.846 [0.815,0.876] | 100.0 | 97.7 | 31.5 | 94.7 | 104 | 295 | 184 | 86 | 94 | 9/6/4/2 |
| fp8_strict | W8A16 | strict | 0.846 [0.816,0.875] | 100.0 | **98.2** | 23.5 | 94.4 | 109 | 327 | 190 | 85 | **71** | 7/7/3/3 |

audit-25 = neg_unment/subthr_neg/misattr/quote_miss on the tester's 25 reports; rep/min for awq_base is its stage-5
full pass, the others include engine start-up over 481 reports. Per-label and per-language AUC are in
`qa/llm_ab/comparison.md`; only contusion moves outside its CI (see §3). **The A6000 is sm_86 and has no FP8 tensor
cores**: vLLM serves the FP8 checkpoint through `MarlinFP8ScaledMMLinearKernel`, i.e. weight-only W8A16 with
dequantisation - a legitimate rung between int4 and bf16, but not FP8 compute, and slower than int4 marlin.

**Paired gold-cell test** - the decisive measurement. Macro AUC on 58 rows has a +-0.03 CI and the 90-96 % of cells on
which two tables agree only add noise to it, so this restricts to the gold cells where two tables *disagree* and asks
which state is closer to the binary gold label under the §2.1 mapping (a sign test):

| vs awq_base | disagreeing gold cells (of 696) | mean abs err base / table | wins base | wins table | sign-test p |
|---|---|---|---|---|---|
| bf16_base | 30 | 0.342 / 0.277 | 13 | **17** | 0.58 |
| fp8_base | 34 | 0.273 / 0.332 | **21** | 13 | 0.23 |
| awq_strict | 57 | 0.250 / 0.319 | **37** | 20 | 0.03 |
| awq_strict2 | 52 | 0.229 / 0.337 | **35** | 17 | 0.02 |
| fp8_strict | 47 | 0.267 / 0.366 | **30** | 17 | 0.07 |

## 3. What actually differs, and why the rewritten prompt loses

Every difference between all six tables is one axis: **how liberally the model writes `negated` instead of `absent`.**
The strict prompt moved 181 cells `negated`->`absent` against only 33 the other way; bf16 moved 101
`absent`->`negated` against 43. Nothing else moved materially.

That axis has a sign, and it is the opposite of the stage-5 audit's. Of the 52 gold cells where `strict2` differs from
`base`, 32 are `negated`->`absent`, and those are **93.8 % gold-negative** (25 of them OA: med_oa 9, lat_oa 7, pf_oa
9, gold-positive rate 0.000). The host graded "ambiguous or borderline findings as negative to favour specificity"
(research.md §1), so a confident `negated` (target 0.04, weight 1.0) beats `absent`, which for the seven
under-reported keys means the calibrated absent target (0.05-0.32) at **weight 0.1**: suppressing a loose negation
replaces a correct confident negative with a low-weight uncertain cell.

So the tester's "absent<->negated on group statements" pattern is an error **against the report text** but, in the
direction the strict prompt fixes it, a regression **against the image-derived gold**. `strict` also broke its own
group rule: STEP 1 made the model leave the OA labels `absent` whenever a normal statement named no compartment;
`strict2` gives STEP 3 precedence and names the unqualified OA negations, recovering 73 of those cells (abs_vs_neg
368 -> 295) but still losing 35-17. The same mechanism explains the one apparently replicated per-label win, contusion
(+0.05 AUC under all five alternatives): those 9 cells are `absent`/`affirmed` -> `negated` on gold-negative rows.

What the rewrite fixed for free: quoting (`quote_miss` 141 -> 90, verbatim 96.6 -> 97.7 %, audit-25 7 -> 2),
attribution (`misattr` 105 -> 86) and `neg_unment` (127 -> 104). The degeneration rule did not work at any precision
(`subthr_neg` 178 -> 184/189/190): STEP 4 is already in the base prompt and neither a louder instruction nor bf16
makes the model follow it.

## 4. Decision and cost

**Do not run a full 4,407-report pass; do not touch `llm_v1` or `targets/v1`.** The criterion set before the runs was
a gold AUC gain beyond the +-0.03 bootstrap noise *or* a clear reduction in the audited error patterns; neither holds.
Every alternative is within 0.005 of the baseline AUC (inside each other's CI), agrees with it on 94-96 % of cells,
and loses or ties the paired gold test.

Cost, measured: a full 4,407-report pass is AWQ int4 **1.9 h on one GPU**, W8A16 2.8 h on one GPU, bf16 TP=2 2.2 h on
**two** GPUs = 4.4 GPU-h - 2.3x the int4 cost for -0.003 AUC. A replacement `llm_v2` needs two variants plus a merge
(4-9 GPU-h) and would invalidate `targets/v1` under the chains currently training on them. Not justified.

**Carry forward** (cheap, for the `targets v1.1` slot already open in `qa/stage5` §Verdict, not stage 6): keep `base`
as the state rules and add only the *quoting* and *attribution* paragraphs of `strict` - they cut non-verbatim quotes
by 36 % with no effect on the state axis, and `evidence_ok=False` currently caps 1,474 corpus cells at weight 0.3. Do
**not** port STEP 1 or STEP 2. The synovitis/contusion weakness is not a prompt or precision problem but the reporting
bias already modelled by the under-reported weights.

## 5. What exists on disk, and how to reproduce

`data/labels/llm_bf16_{bf16_base,fp8_base,awq_strict,awq_strict2,fp8_strict}.{parquet,csv,jsonl,stats.json,gold.json}`
(481 rows each), `qa/llm_ab/{eval_ids.txt,eval_ids.json,comparison.md,comparison.json,comparison_baseline.md}`. The
Qwen3-32B bf16 and FP8 checkpoints were deleted from `HF_HOME` afterwards (98 GB; ~10 min to re-download).

```bash
python qa/llm_ab/make_eval_subset.py                      # 481 ids, deterministic
# one config per screen: CUDA_VISIBLE_DEVICES=0 for awq/fp8, =0,1 for llm_qwen3_32b_bf16.yaml (TP=2)
CUDA_VISIBLE_DEVICES=0 bash scripts/run_screen.sh ab_fp8 .venv-llm/bin/python qa/llm_ab/run_subset.py \
  --config configs/llm_qwen3_32b_fp8.yaml --variant v1 --tag fp8_base         # [--prompt-rules strict|strict2]
python qa/llm_ab/compare.py --baseline awq_base --out qa/llm_ab/comparison \
  --table awq_base=data/labels/llm_v1_raw.parquet --table bf16_base=data/labels/llm_bf16_bf16_base.parquet \
  --table fp8_base=data/labels/llm_bf16_fp8_base.parquet --table awq_strict=data/labels/llm_bf16_awq_strict.parquet \
  --table awq_strict2=data/labels/llm_bf16_awq_strict2.parquet --table fp8_strict=data/labels/llm_bf16_fp8_strict.parquet
```
