# QA report - stages 2+5 (rules/LLM label extraction -> merged table -> targets v1)

Tester, 2026-09-18. Server `gpu-server2`, project `/raid/sroziewski/sandbox/rsna_kad`, local and server copies of
`labels/*.py` + `metrics.py` byte-identical (md5). CPU only, no GPU, no LLM launched, nothing written under `/home`.
Scripts + logs: `qa/stage5/{verify.py,verify.log,spotcheck.py,spotcheck.txt,resume_test.py,resume_test.log}` (server + local).

## 1. Developer tests - PASS
68 passed / 0 failed: test_labels_schema 7, test_labels_rules 30, test_labels_merge 4, test_labels_prompts 7,
test_targets 12, test_metrics 8 (`CUDA_VISIBLE_DEVICES=`, `-p no:cacheprovider`, `TMPDIR=$PWD/.test_tmp`).

## 2. Gold leakage - PASS (one non-leak deviation in 2e)
- (a) 58 gold ids (train.csv non-empty labels) == `folds_v1.is_gold` == `gold_ids.txt`. Few-shot ids: 27 (9 langs x 3),
  all in train, gold overlap **0**; `draft_ids.txt` 27 ids, gold overlap **0**; few-shots are a subset of the draft ids.
- (b) Few-shot report texts: 0 equal to any gold report text (checked whole-text equality, so template duplicates
  would be caught); the 37-row Turkish template contains 0 gold and 0 few-shot studies. `prompts.py` builds messages
  only from `system_prompt`, the few-shot JSON and the report; no label value or path to train.csv labels in the code.
- (c) `calibration_fold{k}.json`: n_gold_train = 47/47/46/46/46 == my recount of gold rows with fold != k
  (11/11/12/12/12 gold per fold). The `counts` per (label, cell) recomputed from the merged table + train.csv gold of
  the training folds match exactly in all 5 files, and EB (strength 8) + PAVA values match to 1e-9. Every
  `fold{k}.parquet`: 4,349 rows, unique ids, **0 gold rows**, ids == train minus gold.
- (d) `llm_v1.gold.json` is produced by `gold_state_report` -> `label_scores` -> `state_score` (targets.py:27-46);
  no `Calibration` object is involved and the report is written in `cmd_merge` (extract_labels_llm.py:93) before any
  target exists. Confirmed numerically in item 4 (independent sklearn recomputation reproduces 0.8517).
- (e) `labels/llm.py` has no http/requests/urllib/socket; only `vllm.LLM.chat` in-process. BUT the vLLM run itself
  wrote under `$HOME` (contradicting the docstring extract_labels_llm.py:13): `~/.config/vllm/usage_stats.json`
  (10.7 KB, hardware/model-config fields only, no report text), `~/.cache/flashinfer`, `~/.tilelang`, `~/.humming`
  (< 100 KB JIT caches, created 14:33 = first vLLM start). vLLM usage telemetry was not disabled
  (`usage_lib.py:268 client.post(_USAGE_STATS_SERVER)`, no `VLLM_NO_USAGE_STATS`/`DO_NOT_TRACK` anywhere) -> an
  off-server POST of environment stats was attempted. No report text or token leaves the box. Fix: set
  `VLLM_NO_USAGE_STATS=1`, `DO_NOT_TRACK=1`, `FLASHINFER_WORKSPACE_BASE`/`TILELANG_CACHE_DIR` in the script.

## 3. Contracts - PASS
| table | rows/unique | missing cols | extra cols | bad enum / NaN | parse_ok | evidence_ok (non-absent) |
|---|---|---|---|---|---|---|
| rules_v0 | 4407/4407 | none | 12 `{k}_mentioned` (not in §3.2, harmless) | 0 / 0 | 4407 | n/a |
| llm_v1_raw | 4407/4407 | none | 12 `{k}_evidence_ok` | 0 / 0 | 4407 (100 %) | 36648/37748 = 97.09 % |
| llm_v2_raw | 4407/4407 | none | 12 `{k}_evidence_ok` | 0 / 0 | 4404 (99.93 %) | 37218/38381 = 96.97 % |
| llm_v1 (merged) | 4407/4407 | none | 84 = 7/key (§3.2 lists 3/key; +state_v2, severity_v2, evidence_v2, rules_state) | 0 / 0 | 4407 | 36274/37748 = 96.10 % |
All state/severity columns are pandas `category` with exactly STATES/SEVERITIES; negated/absent always severity=none,
absent always empty evidence. jsonl: 4407 records each, unique ids == train, 0 torn lines, finish=length 3 (V2).
Targets v1: 5 folds x 4,349 rows x 25 cols (id + 12 target + 12 weight), float32, 0 NaN, min 0.009 max 1.000.
Weights recomputed for **all 4,349 x 12 cells of fold 0** from the merged table with the §2.1 rules
(affirmed/negated 1.0, hedged 0.7, absent 0.5 / 0.1 under-reported, disagreement x0.5, evidence_ok False -> min(w,0.3),
parse_fail/postoperative -> 0.2): **0 target mismatches, 0 weight mismatches** (200-row seeded sample is a subset;
its weight histogram: 1.0 x1440, 0.1 x509, 0.5 x232, 0.05 x101, 0.3 x64, 0.2 x36, 0.7 x16, 0.25 x2).
Monotone affirmed >= hedged >= absent >= negated holds for 12 labels in all 5 jsons (1-2 exact ties per fold from PAVA).
Targets v1 differ from v0 in 99.2 % of cells (stage-6 criterion >= 30 %).

## 4. Gold-58 AUC recomputed with sklearn (§2.1 mapping, 1,000 bootstrap resamples, seed 0) - PASS
| table | macro | CI95 | reported |
|---|---|---|---|
| rules_v0 | 0.8157 | [0.777, 0.854] | 0.8157 |
| llm_v1_raw | 0.8496 | [0.820, 0.880] | 0.8496 |
| llm_v2_raw | 0.8383 | [0.804, 0.873] | 0.8383 |
| llm_v1 merged, primary states only | 0.8496 | [0.820, 0.880] | - |
| llm_v1 merged, V1/V2 score average (= developer's `label_scores`) | **0.8517** | **[0.818, 0.884]** | 0.8517 [0.818, 0.884] |
`state_score` equals my independent mapping on all 4x5x12 (state, severity, key) combinations. Per label (merged avg):
acl .892 mcl .965 med_men .931 lat_men .845 med_oa .887 lat_oa .836 pf_oa .896 effusion .811 synovitis .674 baker .907
contusion .751 fracture .826. Per language: en .859 (n=28), es .875 (10), tr .740 (6), hr .890 (4), bg .750 (3),
el .722 (3), de .833 (2), nl 1.0 (2). Acceptance >= 0.85 met (0.852), goal 0.89 not met; CI is wide (+-0.033).

## 5. Quality spot-check (25 random non-gold reports: en 6, es 4, tr 3, de 3, nl 2, fr 2, hr 2, el 2, bg 1; V1 states)
300 cells: **24 clear state errors (8.0 %)**, 2 severity misuses (0.7 %), 6 non-verbatim/hallucinated quotes (2.0 %,
of which 2 fully fabricated sentences), 3 misattributed quotes, 2 negated cells with empty quote. Error types:
14 absent<->negated confusions on group statements ("cartilage normal", "bones normal") or negated with no mention;
4 "degeneration/mucoid change -> negated" (violates the prompt's own rule -> should be affirmed/mild);
6 wrong-structure attributions (marrow oedema -> OA, trochlea -> lat_oa). 13/25 reports had 0 errors.
Examples (study suffix = last 8 digits of the StudyInstanceUID):
1. `...48083612` (tr) contusion: "Patellar kemikte inferiorda meduller kemikte fokal kemik iliği ödemi izlendi" -> V1 absent (V2 affirmed/mild).
2. `...14972176` (nl) lat_men: the report only says "Kruisbanden en laterale gewrichtspan intact" (lateral meniscus never mentioned) -> V1 negated with the non-verbatim quote "Laterale gewrichtspan intact."; same row: mcl, med_oa, contusion negated without any mention (med_oa/contusion with empty quote).
3. `...83701141` (hr) contusion + fracture: quote "Ostale prikazane koštane, mekotkivne i neurovaskularne strukture uredne su morfologije" does not occur in the report (fuzzy ratio 0.39, fabricated); fracture is never mentioned -> negated.
4. `...04002971` (de) med_men: "Gering mucoide Degeneration des Innenmeniskushinterhornes" -> V1 negated (prompt: affirmed/mild); lat_men "Minimale Signalanhebung ... a.e. bei mucoider Degeneration" -> negated too.
5. `...29634662` (de) lat_oa: "Fokal mäßige Chondropathie an der Trochlea femoris" (patellofemoral) -> V1 lat_oa affirmed/moderate (V2 absent); pf_oa given severity mild instead of moderate.
Corpus-level: merged evidence_ok=False on 1,474 non-absent cells (3.9 %); 424 of them have a verbatim V1 quote and are
flagged only because V2's quote failed; primary-quote misses are 855 negated / 188 affirmed / 7 hedged, mostly stitched
paraphrases of group-negation sentences (ratio 0.6-0.78) - the 0.8 threshold is doing its job; fabrications are rare.

## 6. Consistency - PASS
V1==V2 state agreement per label: acl .953 mcl .941 med_men .969 lat_men .963 med_oa .849 lat_oa .844 pf_oa .910
effusion .973 synovitis .988 baker .941 contusion .865 fracture .830 (cell-level incl. severity: 3-18 % disagreement).
V1 vs rules (hedged->affirmed, all rows / both mentioned): acl .84/.91 mcl .85/.90 med_men .86/.92 lat_men .81/.86
med_oa .69/.88 lat_oa .69/.87 pf_oa .77/.92 effusion .95/.97 synovitis .97/.99 baker .90/.96 contusion .75/.89
fracture .74/.97; mention_conflict rate 0.0-0.25 (fracture highest: rules fire on "no fracture" sentences the LLM leaves absent).
Disagreement -> targets: 20 random disagreeing fold-0 cells all equal mean(cal[v1 cell], cal[v2 cell]) and weight x0.5
(e.g. affirmed_mild vs absent pf_oa -> 0.236/w0.50; absent vs negated fracture -> 0.118/w0.05). 5,381 fold-0 cells (10 %)
are disagreements. Turkish template: 37 identical reports, **one** state signature per table (rules, V1, V2, merged):
10 negated + 2 absent, 0 affirmed; 1 distinct target row in fold 0. (V1 sets normal_template on 0/37, V2 on 37/37;
merged = AND -> 0/37; informational flag only.)

## 7. Under-reporting - PASS
affirmed rate vs gold prevalence: synovitis 12.5 % vs 46.6 %, fracture 5.1 % vs 31.0 %, contusion 23.0 % vs 32.8 %,
baker 24.3 % vs 20.7 %, med_oa 41.3 % vs 25.9 %, lat_oa 26.3 % vs 19.0 %, pf_oa 44.6 % vs 36.2 %, acl 24.6 % vs 41.4 %,
lat_men 23.4 % vs 39.7 %, mcl 17.4 % vs 15.5 %, med_men 50.3 % vs 44.8 %, effusion 61.0 % vs 60.3 %.
Absent weight = 0.1 for synovitis/fracture/contusion/baker/med_oa/lat_oa/pf_oa (config == `UNDER_REPORTED_KEYS`), 0.5 else.
Calibrated absent target per fold 0-4: synovitis .290/.298/.314/.295/.322 (gold absent cell fold0: 34 rows, 11 positive);
fracture .139/.115/.119/.089/.088; contusion .072/.116/.116/.113/.110; baker .063/.067/.071/.039/.065;
med_oa .096/.092/.110/.088/.050; lat_oa .085/.110/.139/.076/.110; pf_oa .110/.152/.247/.191/.210.
Note: fracture hedged is calibrated to 0.600 in 4 folds -> those 121 fold-0 cells sit exactly on POS_THR and count as positives.

## 8. Metrics - PASS
float32 targets [0.2, 0.6] with weights [0.5, 1.0] -> keep [T, T], pos [F, T] (0.2 negative, 0.6 positive, as §2.5 says);
0.2f32 = 0.2000000030 and 0.6f32 = 0.6000000238 are absorbed by `THR_EPS=1e-6` (metrics.py:14,45-46); weight 0.5f32 kept,
0.49 dropped. Test exists: `test_binarise_float32_targets_exactly_on_thresholds`. `macro_auc` on a random 500x12 case
== sklearn `roc_auc_score` per label to 0.0e+00. Fold-0 real targets: 0 cells at 0.2, 121 at 0.6 (see §7).

## 9. Code review (correctness)
- **BUG-1 (low, 3 rows)** merge.py:109-111 + targets.py:237-244: when V2 fails to parse (3 Greek studies, ids
  `...7086010772 ...2250214590 ...7272730103`), its default all-absent states are stored as `{k}_state_v2` and
  `make_targets` treats every cell as a voter disagreement: targets are averaged with the absent value and weights
  halved (measured: acl affirmed -> 0.47/w0.5 instead of 0.88/w1.0; reproduced synthetically in resume_test.py).
  Fix: copy the surviving voter's states into `_v2` (or skip disagreement) when `ok2` is False; the `evidence_ok`
  branch already does this (`o1 & (o2 | ~ok2)`).
- **BUG-2 (low)** llm.py:196-207: the jsonl is opened in append mode without checking that it ends with "\n"; a torn last
  line (crash mid-write) makes the next record land on the same line, and `_read_jsonl` drops both on the next resume
  (that study is simply re-run; no wrong data). Fix: write "\n" first if the file does not end with one.
- Resume verified with a fake LLM: crash after batch 1 -> 2 records; resume runs the remaining 5, table 7 rows/7 unique
  ids, duplicate id line -> last record wins, `retry_failed` re-runs only the 1 truncated study (n_run 2 incl. the torn
  one), no-op resume makes no LLM call. Real jsonls: 4407 unique ids, 0 duplicates, 0 torn lines.
- Parse failures: kept as rows with `parse_ok=False`, absent states, weight 0.2 (targets.py:229); merged `parse_ok = ok1|ok2`.
- PAVA (targets.py:123-136): hand cases [0.9,0.5,0.6,0.1] -> [0.9,0.55,0.55,0.1], [0.1,0.9] w[1,3] -> [0.7,0.7],
  [0.9,0.2,0.5,0.4] w[1,1,3,1] -> [0.9,0.425,0.425,0.4]; 200 random weighted 4-chains == sklearn IsotonicRegression(decreasing).
- Categorical -> NaN: `finalize_label_table` (schema.py:160-163) would silently turn any out-of-enum value into NaN and
  `write_label_table` does not call `validate_label_table` (only extract_labels_rules.py does). Guarded upstream by
  `parse_output`'s enum checks; measured 0 NaN in all 4 tables. Recommend calling `validate_label_table` in `write_label_table`.
- Design notes (not bugs): merged `evidence_ok` is False when either voter's quote fails (424 cells lose weight although
  the primary quote is verbatim); `retry_failed` rewrites the jsonl with mode "w" (a crash during the rewrite loses it).
- Acceptance-row deviations: V2 full pass 16,152 s (269 min) > the "< 2 h per variant" criterion (V1: 98 min); V1 first
  pass truncated 78 / V2 77 outputs at max_tokens 900, re-run at 1,400 -> final truncation 0 / 3.

## Verdict
**stage 2+5 ACCEPTED** (all acceptance numbers met: rules 0.816 >= 0.75, merged gold 0.852 >= 0.85, parse 100 %/99.93 %,
evidence 97.1 %/97.0 %, 4,407/4,349 rows, no gold leakage). Follow-ups for the developer, none blocking stage 6:
BUG-1 (V2 parse-fail counted as disagreement, 3 rows), BUG-2 (jsonl newline guard), disable vLLM telemetry and redirect
JIT caches away from `$HOME` before any rerun, V2 prompt relayout for prefix caching (269 min), and the two systematic
LLM patterns from §5 (degeneration -> negated; negated on unmentioned structures) as candidates for a prompt fix in stage 9.