# Audit 2026-09-25 - 6. REVIEWER: adversarial review of the plan (written 2026-09-26)

Inputs: audits 1-5, architecture.md §2 + §4.1-§4.3, arbiter/report.md, competitor/analysis.md, qa/pseudo_v3/*. Server
read-only except scratch `qa/audit_2026-09-25/reviewer/` (`r1_noise.py/.md` = paired LB/gold noise; `r2_checks.py/.md`
= arm marginals, optimum set, hedge simulation). Server state read 09-25 23:36 CEST. Nothing trained or submitted.

## Verdicts at a glance

| # | Question | Verdict |
|---|---|---|
| 1 | Statistical validity of G1-G3 and gold-58 | **Flawed.** The noise is manageable (paired SDs below); the flaw is that G1 measures the wrong thing, and G2 has no threshold. |
| 2 | Pseudo-label bet | **Design sound, judging flawed.** Leakage is argued correctly. Consensus cannot judge this lever. The likely false positive is not caught by G1. |
| 3 | Final selection | **Direction sound, rationale overstated.** The hedge is worth ~0.0002-0.001 AUC. Dropping efficiency eligibility is right, but then bet 3 is wasted work. |
| 4 | Missing / overweighted | **Flawed.** Training compute is ~85 % idle while inference runtime is the binding constraint. Full-data retraining is missing, a free routing fix is missing, and the Claude lane contradicts §4.3. |
| 5 | Process risks | **Mostly aspirational.** No git yet, main_v3 runtime still unread, the calendar is already wrong about what the GPUs are running, and quota is shared. |
| 6 | Integrity of our conclusions | "A+E+C honest optimum" **holds**. "More arms saturate" is **right in size, wrong in evidence**. "v1.1 does not move the model" is **unresolved, not negative**. "Labels alone gave +0.0265" is **false as worded**. |

## Q1. Are the gates deciding on noise?

I simulated a public split by subsampling 390 of our OOF studies, and a private split with 910, keeping the pairing
(`r1_noise.md`). A gate has to beat the SD of a paired difference, not the SD of a single score (the level).
"Gold-like" = the 58 gold rows resampled to 390 studies (test-style labels where every cell counts).

| pair (proxy) | level SD, 390 | paired-delta SD, weak / consensus / gold-like (390) | gold-58 paired SD | private (910) paired SD |
|---|---|---|---|---|
| E vs MVP: same arch, targets changed (≈ E_p vs E) | 0.012 | **0.010 / 0.010 / 0.006** | 0.017 | 0.006 |
| A vs A8: near-copy single | 0.010 | 0.005 / 0.006 / 0.003 | 0.008 | 0.003 |
| A+E+C vs A8+E+C: one member swapped | 0.009 | **0.0019 / 0.0024 / 0.0013** | 0.0036 | 0.0009 |
| A+E+C vs A+E: one diverse arm added | 0.009 | 0.0026 / 0.0034 / 0.0016 | 0.0046 | 0.0013 |
| A+E+C vs E: ensemble vs single | 0.009 | 0.0065 / 0.0063 / 0.0029 | 0.0081 | 0.0036 |

- **The level noise (0.0097) is the wrong yardstick for a gate that compares two submissions.** Paired, the public LB
  resolves ~0.005 between correlated ensembles (2 SD) and ~0.012-0.02 between different single models; §4.1
  consequence (2) ("no submissions on A/B < 0.02") is too conservative for paired ensemble probes. The LB rounds to 3
  decimals (±0.0005).
- **G1** (E_p vs eff_v1, promote +0.010, kill < +0.005), paired SD ~0.006-0.010: with no true effect P(promote) =
  5-16 %; with a true +0.010 P(kill) = 20-31 %. Borderline, not pure noise. **The real defect is what G1 measures
  (Q2).** E_p also changes targets v1→v1_1, a small extra confound.
- **G2** ("main_v4 if LB ≥ 0.933") is a 0-SD threshold, a coin flip under no effect (paired SD ~0.0025-0.004).
  **G3** (+0.010 between round-2 and round-1 students, paired SD ~0.004-0.006): sound, slightly conservative.
- **Gold-58 is paired too, and resolves better than "±0.027" implies for correlated pairs.** The ±0.027 is the
  *median* paired half-width across very different models; between two ensembles it is ±0.007-0.009. Gold-58 and
  the public LB are independent samples, so pool them: an inverse-variance Δ with σ_LB ≈ 0.0025 and σ_gold ≈ 0.0045
  gives σ ≈ 0.0022 for an ensemble-vs-ensemble decision.

## Q2. The pseudo-label bet

- **Construction is sound.** Verified in `make_targets_v3.py`: fold-k own rows bit-identical to v1_1, gold excluded,
  mix formula and weight monotonicity asserted. Do not tune the pre-registered 0.5/0.5 mix on gold-58 (overfitting),
  even though the mix is below the teacher on synovitis (0.799 vs 0.824) and effusion (0.940 vs 0.965).
- **Leakage argument: correct.** No gold row reaches any teacher or student. The residual channel: a fold-k student
  distils teachers M_j (j≠k) fitted on fold-k *weak* rows and sites. It never sees fold-k images, so the leak travels
  only through look-alike scanners in other folds - small but unmeasured, and larger for weak/consensus (the teacher
  fitted exactly those labels) than for gold. The public LB is the only fully clean judge.
- **Consensus is not a valid judge of this lever.** It keeps only unanimous affirmed/negated cells, so it is blind to
  the silent cells where the headroom lies (2,960 synovitis and 788 pf_oa silent cells re-weighted); it counts
  "affirmed-mild" cells as positive although the image-definition shift should push them down (§4.1 LLM A/B: 93.8 %
  of those are gold-negative); and it is flattered by the teacher leak. Lenient harm check only, never a PROMOTE clause.
- **The false positive G1 invites.** E_p's teacher is A+E+C, which beats E by +0.026 weak and +0.038 gold
  (`r1_noise.md`); a student distilled from a much stronger teacher gains **even if the pseudo-label mechanism adds
  nothing**. The v3 targets score 0.915 on gold vs the teacher's 0.904 - essentially the teacher (+0.011). So E_p can
  pass G1 on distillation alone; main_v4 (the teacher distilled into its own three recipes) then lands at main_v3 ±
  0.005, and G2's "≥ 0.933" accepts that tie. Score damage is nil (main_v3 stays in the final two). The waste is
  ~25 GPU-h plus the round-2 lane, which rests on the same confounded single-model gate (G3 again compares single E
  students). **The plan does not catch this before 10-21**: the only ensemble-level test (G2) has no threshold.
- **Fix.** G1 becomes kill-only: stop the lane if E_p − eff_v1 < +0.005 on the LB AND gold Δ < +0.01. Train A_p
  and C_p unconditionally (the GPUs are idle, Q4, so gating GPU spend buys nothing). The only decision that matters is
  main_v4 vs main_v3 on the pooled paired Δ (LB + gold-58): accept at ≥ +0.004 (≈ 1.8 σ) with gold Δ ≥ −0.009.
  Round 2 runs only if main_v4 passes that test, not on a single-student G3.

## Q3. Final selection

- **Giving up efficiency eligibility is right:** rank 478; ranks 1-10 carry public 0.945-0.958 in minutes; our 0.913
  is ~0.035 AUC ≈ 40 Kaggle-minutes-equivalent behind.
- **A second main pick is worth little, but keep it (> 0, free)** (`r2_checks.md`): at a private paired SD of
  ~0.0015-0.0025, E[max of two] − E[the one picked by the paired public LB] = **+0.0002 to +0.001 AUC** (true-delta
  uncertainty τ = 0-0.010), i.e. ~30-140 public ranks near 0.933 (~140 teams per 0.001) - not a medal lever.
- **Missed consequence: bet 3 (decode engineering, 1.5-2 dev-days, 3-4 submissions) buys nothing** once slot 2 goes to
  main. Cut it, keep E_p as a probe only, delete the "efficiency if in the prize zone on 10-18" branch.

## Q4. Missing / overweighted

1. **Compute is not the constraint; Kaggle runtime is.** 2 A6000 × 23 days to the 10-18 freeze ≈ 1,100 GPU-h; the
   plan uses ~100-150; all work to date used 53. Members are capped by the 9 h runtime (main_v3 projected 3.4-5.8 h,
   **never measured**), so the value is in stronger members at equal cost: **full-data retrains** (last-epoch policy
   needs no validation fold; +25 % data; past winners did it), e.g. **3 full-data seeds per arm instead of 5 folds**,
   which also frees runtime for a 4th diverse arm. A paired LB probe vs the 5-fold twin (SD ≈ 0.002-0.003) catches a
   mis-train; pseudo targets already exist for all 4,349 rows.
2. **"More arms are saturated" rests on the wrong evidence** (Q6); the STOP holds on size (+0.001-0.003), not on it.
3. **Free test-shift hardening is missing.** `series.assign_slots` routes on `Fluid_Sensitive` only (`series.py:42`).
   In train, FS == FatSup for all 24,371 series; the host says test differs. A fluid-sensitive series without fat
   suppression (e.g. T2 non-FS) can win the `*_fs` slot on slice count and feed the model an image type it never saw.
   Fix: `Fat_Suppression` as the first rank key inside the FS slots - a no-op on train (no retrain, OOF unchanged) plus
   one test. The prior-row alarm cannot see mis-routing, and "main_v3 already ran the hidden set" proves it ran, not
   that it routed well. Prevalence shift is correctly ignored (macro AUC is invariant); test-time adaptation: skip.
4. **The Claude voter lane (bet 2) contradicts §4.3.** Its pilot gate is a *label* gain ≥ +0.010, but a +0.015 label
   gain did not visibly move the model, so passing predicts nothing - and it burns the quota every agent shares. Drop
   it, or use a paid API key (allowed, host 733965; forum cost < $5), never the Max-plan quota.
5. **STOP list.** Correct: no re-blend, no LB-tuned weights, no confident learning, no cache v2, freeze after 10-15.
   "No seeds" is wrong only where seeds replace folds (item 1). "No RadImageNet" holds on runtime, not saturation.

## Q5. Process risks: real or aspirational?

**Aspirational today:**
- **No VCS** (no `.git` on the server at 09-25 23:36). Snapshot and export guard must *block* the 09-27 E_p export.
- **main_v3 runtime unread 3 days after scoring** - a 5-minute UI read that every runtime rule depends on.
- **The calendar is already wrong** (G1 on 09-27 still fits). Arm D's chain has no `FOLDS` env, so it runs **all 5
  folds** on GPU 1 (~12 h), not fold 0 ending ~09:00. Arm N triage runs 4 configs in sequence on GPU 0 at 142 img/s
  (~14 h), not ending ~07:30. E_p runs 5 folds **on GPU 0 only** at **128 img/s vs 470 standalone** (GPU 0 also hosts
  another user's 8-process job), landing ~09-26 afternoon, not split across GPUs.
- **Chain depth ≥ 24 h is not implemented:** the `pseudo` chain ends at `predict_oof` with no successor; no kill-switch.
- **The quota cap** ("≤ 2 pilot-equivalents/day") is enforced by the agents that consume the quota - not a control.
  Moving the Claude pass off the Max plan (Q4.4) is.
**Real:** the new-dataset-slug rule; the kernel-vs-local record (1.78e-4); main_v3 scored on the full hidden set.

## Q6. Re-verification of load-bearing conclusions (raw OOF, `r2_checks.md`)

- **"A+E+C is the honest optimum": HOLDS.** A+A8+E+C+MVP − A+E+C = +0.0024 [+0.0009, +0.0040] weak, +0.0019
  consensus, +0.0034 [−0.0055, +0.0116] gold - a tie that costs 10 more members. Every subset that drops an arm loses
  on all three arbiters (e.g. A+C −0.0022 / −0.0040 / −0.0031). Per-arm equal weight holds (5/5/5).
- **"More arms saturate (4th +0.0007)": evidence does not support the claim as worded.** The only 4th arms tried are
  A8, a near-copy of A (ρ 0.950), and MVP, a weak model (ρ 0.72-0.76). Strong diverse arms gave 2nd +0.0105 weak /
  +0.0109 gold (A→A+C), 3rd +0.0022-0.0050; extrapolated, a 4th is worth ~+0.001-0.003. "Cannot reach 0.943 with
  arms" stands; "saturated" should read "diminishing, ~+0.002 per further diverse arm, below resolution".
- **"Targets v1.1 does not move the model": UNRESOLVED.** One fold: 869 weak studies, 11 gold rows; gold 0.8530 vs
  0.8985 (verified in both `fold0/metrics.json`) is noise. Paired consensus SD at n≈880 is ~0.004-0.005, so consensus
  +0.0022 / strict +0.0054 cannot exclude +0.005. Correct reading: "no effect ≥ ~0.01" - still enough to not retrain.
- **§4.1 "0.8815 → 0.9080 purely by swapping rule targets for LLM targets": FALSE AS WORDED.** The configs also changed
  epochs 6 → 8 and `select_epoch` best → last (`mvp_effv2s_256.yaml` vs `eff_effv2s_256.yaml`), and 0.8815 is MVP
  rescored on the v1 targets E trained on (yardstick bias). "Labels are the lever" survives on the LB (0.865 → 0.913).
- **Minor:** the (A+E) vs A LB "inversion" is −0.0097 against the OOF expectation at paired SD 0.0044 ≈ 2.2 σ -
  borderline evidence that weak-OOF over-credits E vs image labels. Diagnosis #5 (leaders +0.03-0.04 on gold-58) uses
  [S] numbers from teams that may train on gold (one public fork weights gold 3.0 and picks epochs on it): not evidence.

## Required changes (the plan is not fit to execute without these; all are cheap)

1. **G1 becomes kill-only** (Q2); train A_p and C_p unconditionally.
2. **main_v4 vs main_v3 on the pooled paired Δ (LB + gold-58):** accept at ≥ +0.004 with gold Δ ≥ −0.009. Round 2 is
   gated on this, not on single-student G3.
3. **Consensus out of every PROMOTE clause** for pseudo or new-target runs; harm check only.
4. **Read the main_v3 and eff_v1 hidden runtimes today** and log them in §4.1; select 56488491 in the UI + screenshot.
5. **Git snapshot, `RSNA_KAD_GIT_SHA` and the export guard land and are tested before the E_p export.** No exceptions.
6. **Cut bet 3's decode engineering** and the efficiency-slot branch. 7. **Claude lane: drop, or paid API key only.**
8. **Re-baseline the calendar on the real GPU state** (arm D on 5 folds - set `FOLDS` or accept it; E_p on a contended
   GPU 0) and queue a named successor behind every chain (default: full-data retrain of C on v3_pseudo).

## Recommended

Full-data retrains (e.g. 3 seeds per arm) as final members, each validated by a paired LB probe against its 5-fold
twin · a `Fat_Suppression` tie-break in slot routing plus one test · correct the §4.3 saturation/v1.1 wording and the
§4.1 "purely by targets" line · spend submissions on paired probes, the only clean judge (resubmitting an identical
entry never reduces noise) · RadImageNet fold-0 only as GPU-idle filler, after the runtime is known.

## Overall verdict

**Not fit to execute as-is; fit after the 8 required changes, most costing hours.** The core is right (one
pseudo-label round, stop same-kind arms, equal weights, no LB tuning, main_v3 as anchor), and score risk is low since
main_v3 stays in the final two. The risks are wasted days: a confounded G1 that will likely "pass", a G2 without a
threshold, 2 dev-days on an unselected efficiency entry, a quota-burning label lane our own evidence says cannot move
the model, and process mitigations that are still plans.
