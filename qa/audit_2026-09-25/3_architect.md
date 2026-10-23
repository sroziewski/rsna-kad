# Audit 2026-09-25 - 3. ARCHITECT: the plan for the last 26 days (09-26 -> 10-22)

Inputs: 1_researcher, 2_analyst, 4_developer, 5_tester, docs/architecture.md §4.1-§4.3, qa/competitor/analysis.md,
qa/pseudo_v3/*. Planning only. Tags as in the researcher report: [V] verified, [S] second-hand, [U] our guess.

## 1. Diagnosis (why 0.933, while forks sit at 0.943 and the top at ~0.959)

1. The shipped blend scores gold 0.904, but its training labels score only 0.834 against gold. **The model beats its labels by +0.07**, so we are fitting targets that are worse than our model.
2. 71 % of the gold shortfall (0.068 of 0.096) sits in 5 labels (lat_oa, synovitis, pf_oa, lat_men, contusion). They are 0.87-0.96 on weak labels and 0.06-0.13 lower on gold. The loss comes from the gap between what reports call positive and what the host's readers call positive, not from fitting the reports.
3. Label polish does not close that gap. v1.1 improved the labels by +0.015 and moved the model by < 0.005 (§4.3), the forum saw the same, and even unanimous voters are wrong 19 % of the time.
4. The credited lever does close it. Three top-50 reports [S] and the RSNA-2024 lumbar 2nd place [V] used a teacher -> 0.5 label + 0.5 teacher soft target on ALL cells -> student. This moves targets from the report definition toward the image definition. Stage 9 rejected it on weak-OOF, which by construction cannot see this lever.
5. Our single models trail the leaders by ~0.03-0.04 on the same gold-58 (0.89 vs ~0.93). That is the same size as the LB gap to 0.95-0.96, and those leaders reached it with ONE small model at 224-288 px. Resolution is not the driver.
6. Adding arms has saturated: the 2nd arm gave +0.0077, the 3rd +0.0049, the 4th +0.0007, and the best re-blend is a tie (+0.0024). More arms of the same kind cannot reach 0.943.
7. The 0.943 pile-up is LB-tuned forks of third-party weights (§competitor). Its real signal is ~0.935-0.94; the rest is public-split fit. It is not a target to copy, only a shake-up candidate.
8. The residual image-limited loss is the lateral compartment (lat_oa -0.032 below its label ceiling, mcl -0.018, pf_oa -0.015), worth at most ~0.005 macro.
9. Resolution limits: offline arbiters resolve ~0.005 and the public LB ~0.02. Only bets that can plausibly move the score by >= 0.01 are worth making, and each needs a pre-registered gate.
10. Verdict: **the gap is in the targets, not the network.** Bet on pseudo-labels, stop building arms, and harden what we ship.

## 2. Strategy - ranked bets

**Judging rule for every bet below (amends §4.2).** Weak-OOF is contaminated for pseudo-label and new-target runs. Every
teacher OOF prediction on a fold-j study came from a model trained on fold k's weak rows, so the student's fold-k OOF has
seen fold-k labels through the teacher. Weak-OOF is printed but never decides. The arbiters are:
- **consensus**: voter-code based, the same yardstick across target versions.
- **gold-58**: never trained on. It is near-clean only: a second-order site leak through the teacher is possible, so it is a veto plus a +/-0.02 signal.
- **a pre-registered paired LB probe**: the only fully clean judge. It is the same recipe with only the targets changed, against an already-scored twin.

| # | Bet | Mechanism / evidence | Cost | Gate (pre-registered) |
|---|---|---|---|---|
| **1** | **Pseudo-label round 1 -> `main_v4` = A_p + E_p + C_p** (the same 3 recipes and 15 members as main_v3, retrained on `v3_pseudo`) | Diagnosis 2-5. Headroom: 3,700 silent synovitis cells and 985 pf_oa cells (stage 9c). Forecast +0.005..0.015 LB [U]. | E_p 4.5 GPU-h (running). C_p ~10, A_p (8 ep) ~10.4. **~25 GPU-h, ~3 calendar days**, 0 Claude quota. Kaggle runtime unchanged vs v3. | **G1 (09-28)**, probe = E_p vs eff_v1 (0.913), same export config. PROMOTE if LB >= 0.923 and gold >= -0.02 and consensus >= -0.005, or if gold >= +0.02 and LB >= 0.913. KILL if LB < 0.918 and gold < +0.01. In between: finish C_p only (already speculative) and let the G2 probe decide. **G2 (10-01)**: main_v4 LB probe vs 0.933. |
| **2** | **Pseudo round 2 on the 5 gap labels**: teacher = main_v4 OOF; the label side gains a 5th voter from the Claude pass (host thresholds in the prompt) | Raymond's report: 0.940 -> 0.949 over repeated teacher rounds [S]. The Claude voter aims at the definition gap (host severity thresholds), not at polish. Weakest evidence of the four bets. | Claude pass on 3,926 reports ≈ 8x pilot quota. E round-2 4.5 GPU-h, then C + A ~20 GPU-h. | **Only if G1 promoted.** Pilot gate (09-27): the 5-voter merge on 481 studies / gold-57 reaches **>= +0.010, P >= 0.95** over the 4-voter merge. Otherwise the lane closes and round 2 runs without the Claude voter. **G3 (10-08)**: E round-2 vs E_p paired LB >= +0.010, else STOP round 2. |
| **3** | **Efficiency entry = E_p student + decode engineering** | eff_v1 is prep-bound: 7.2 of 9.2 A6000-min go to decode. The 743374 list attacks exactly that (16-thread reads, raw header parse, strided percentiles, fp16 without autocast) [S]. **1 min ≈ 0.00083 AUC.** | 1.5-2 dev-days, 0 new GPU-h (reuses E_p), 3-4 submissions | Keep the lowest measured score = `-AUC/0.45 + runtime_s/32400`, with runtime read from the Kaggle UI. 2 vs 5 folds is decided on this score, not on AUC alone. |
| **4** | **Arm N / arm D (in flight): finish, judge, and most likely stop** | Per-label heads are the one structural feature shared by 0.94x readers [V code]. Arm D targets lat_oa/mcl. Forecast value ≤ +0.005 (analyst #3/#4). | Sunk. A 5-fold follow-up costs ~11 GPU-h. | Fold-0 A+E+C+X vs A+E+C on **consensus** (different targets): >= +0.010, with gold not vetoed. Fold 0 overstates by ~2x (§4.3), so this bar is deliberately high. On a pass, retrain on **pseudo targets** as a *replacement* for the weakest LOO arm, and add it only if the measured runtime allows. Decided 09-27. |

**STOP (decided):**
- Stop new backbone arms and seeds of the current arms before bet 1 resolves.
- Do not re-blend existing arms (A+A8+E+C+MVP is a tie).
- Do not run stage-9-style leave-2-out honest pseudo-labels or confident learning.
- Do not do label polish for its own sake (including a second Claude pass or prompt iteration).
- Drop laterality canonicalisation, a larger K, per-label or LB-tuned weights, and a RadImageNet/DINOv3 member (no Kaggle-runtime room, and synovitis already beats its labels).
- Drop the VLM (Qwen3.5-2B) bet and any cache v2.
- No team merger unless an offer with self-trained, releasable weights is in hand by 10-12.
- After 10-15, no new idea of any kind.

## 3. Final-selection plan (select on 10-21, a day early; the deadline 10-22 is 23:59 UTC)

- **Slot 1 = the main entry.** Pick main_v4 (or its round-2 successor) if its LB >= 0.933 and gold/consensus of the
  assembled blend are >= main_v3 - 0.005 (no veto). Otherwise pick **main_v3 (56488491)**. It is scored, safe
  and verified; select it in the UI now as insurance.
- **Slot 2: this is the real trade-off, decided explicitly.** Efficiency eligibility requires the efficiency entry to be one
  of the 2 picks. We sit at efficiency rank 478, and ranks 1-10 carry public 0.945-0.958 in minutes, so the realistic
  prize probability is ~0. A second main pick is a real hedge against private shake-up, because Kaggle takes the better
  of the two private scores. **Rule: slot 2 = the efficiency entry only if on 10-18 it ranks inside the efficiency prize zone
  (number of prizes: check the Efficiency Prize page, [U]). Otherwise slot 2 = the other main candidate: robust
  main_v3 plus risky main_v4.** If bet 1 was killed there is nothing to hedge with, so slot 2 = the efficiency entry (free).
  The coordinator/user may overrule this in favour of efficiency eligibility; the cost is losing the main hedge.
- **Robust vs risky, when only one main slot is left:** take the risky entry only if BOTH hold: an independent clean signal
  (the paired LB probe of its E-student twin >= +0.010) AND no offline veto (gold >= -0.02, consensus >= -0.005).
  Otherwise take the robust one. Never decide on a public-LB delta < 0.02 alone.
- **Kaggle-runtime budget.** main_v3 = 15 members, projected 3.4-5.8 h. The real number is still owed. §2.7 rule:
  projected x1.25 < 6 h. If the real runtime R <= 3.4 h, there is room for +1 A-size arm (~1.2 h). If R >= 4.5 h, there
  is no room, and anything new must *replace* members. main_v4 has the same 15-member shape, so it needs no new budget.
- **Measurements owed, all by 09-27:**
  - (a) main_v3 and eff_v1 hidden runtime, read from the submission pages in the UI (5 min, coordinator).
  - (b) Value of z-TTA: OOF pass of A/E/C fold models with and without the z-offset on the cache (~3 GPU-h, GPU 1). If the gain is < 0.002 on weak and consensus, drop z-TTA; that frees ~1.7 h for one more member or a seed.
  - (c) The real E_p and final-eff runtime from the UI after each probe.

## 4. Calendar (G0 = GPU 0, G1 = GPU 1, CQ = Claude-quota lane; every GPU line is a queued screen chain >= 24 h deep)

| Date | G0 | G1 | CQ / coordinator | Checkpoint |
|---|---|---|---|---|
| 09-26 | arm N triage ends ~07:30 -> E_p folds (split with G1) | arm D ends ~09:00 -> E_p folds -> z-TTA OOF pass | Claude pilot finishes and is scored. Read runtimes (UI). **Select 56488491 in the UI.** | N/D fold-0 judged once their screens are dead |
| 09-27 | speculative C_p folds 0,2,4 | speculative C_p folds 1,3 | Pilot gate (bet 2). Dev: export guard + git snapshot land **before** the E_p export. Export E_p to a **new dataset slug**, verify kernel vs local, **LB probe #1** | |
| 09-28 | A_p (8 ep) folds 0,2,4 | A_p folds 1,3 | if the pilot passed: Claude full pass starts (≤ 2 pilot-equivalents/day) | **G1**: promote/kill bet 1 (kill = stop A_p, finish C_p) |
| 09-29/30 | A_p finishes; guarded OOF assembly | eff decode bench (A6000) | main_v4 export, verify, **LB probe #2** (09-30) | |
| **10-01** | round-2 E student if the targets are ready, else C_p seed 2 (speculative) | arm N/D 5-fold on pseudo targets if promoted, else C_p seed 2 | Claude pass continues | **G2**: main_v4 accepted / rejected |
| 10-02..04 | as above | as above | Claude pass **hard stop 10-04** (partial = dropped). Dev: inference robustness, P1 items | round-2 targets built 10-04 (teacher = main_v4 OOF, + Claude voter if complete) |
| 10-05 | E round-2 student (4.5 GPU-h, split) | | **LB probe #3** (E round-2 vs E_p). Eff speed builds: 1-2 submissions | |
| 10-06..08 | C round-2 (if G3 is on track) | A round-2 | | **10-08 G3**: round 2 go/kill. Eff recipe (folds, K) frozen |
| 10-09..14 | round-2 finishes, or seeds/replacement arm (only within the measured runtime) | same | Dev: hardened kernel end-to-end verification on the final members | 10-12: merger decision (default: none) |
| **10-15** | **FREEZE new ideas**; rules re-check (efficiency clause, merger, external-data disclosure) | | team merger/entry deadline | |
| 10-16..18 | reproducibility check: retrain 1 fold from the snapshot (P2) | idle / verification | final main + final eff exports; kernel-vs-local; runtime x1.25 check | **10-18 final ensemble frozen** |
| 10-19..20 | - | - | final submissions (≤ 3 per day kept in reserve). 10-20 = last planned submission | **10-20 final submissions scored** |
| 10-21 | - | - | **select 2 in the UI** (1 day early), screenshot | 10-22 = slack only |

Slack: roughly 1.5 days hidden in 10-09..10-14, plus 10-21/22. If bet 1 is killed on 09-28, G0/G1 run the stopped round-2
lane as seeds/replacement per bet 4 or sit idle; idling is acceptable, but losing insurance is not. Winners' package
10-23..11-05 (see §5).

## 5. Engineering track (owner / by date / effort)

| Item | Source | Owner | Must land by | Effort |
|---|---|---|---|---|
| Export guard: refuse a fold with `last.pt`, `metrics.epoch != epochs-1`, or a live screen; **fail** on a missing fold; assert equal folds per arm (or weight per arm) | dev P0-2, P1-9 | developer (+ tester test) | **09-27, before the E_p export** | 3 h |
| Liveness guard in `compare_ensemble`/`evaluate_runs`/`predict_oof`; `--recompute` writes to a new file | dev P1-6 | developer | 09-29 (main_v4 OOF) | 2 h |
| **Frozen snapshot + VCS**: `git init` on the server (the authoritative tree); commit; tag `main_v3-lb0933`; set `RSNA_KAD_GIT_SHA` in `run_screen.sh`; tag every dataset/kernel push. Drift control: `remote.ps1 sync` only after an md5 diff shows local ⊇ server; the notebook is only regenerated on the server | dev P0-3, B1 | developer | **09-27** | 2 h |
| Inference robustness: per-study `get(timeout)` -> prior row; per-study try/except in the consumer -> prior row; top-level fallback writes a prior-only file; **prior-row alarm** (> 10 % prior rows -> fail loudly: a scored 0.5 file is worse than a visible error); wall guard counts from interpreter start; `spawn` instead of `fork` | dev P0-4, 1.2, 1.4, B2, B3 | developer + tester drills (hang, OOM, decoder missing) | **10-06** (the final kernel is built on it) | 0.5-1 d |
| One cache `params_sha256` across members; cost-balanced `assign_devices` (B4, ~15 % wall) | dev P1-5, B4 | developer | 10-06 | 2 h |
| timm assert/wheel + pinned kernel env; fp16/fp32 gap and kernel-vs-local diff recorded for the final ensemble; `bench_infer` pass = the §2.7 rule (B7) | dev P1-8, B7 | developer / tester | 10-14 | 0.5 d |
| Manifest fix for `v1_1_gemma3`; script the gemma/pseudo/Claude target builds under `scripts/` (out of `qa/`) | dev P1-7 | developer | 10-08 | 3 h |
| Every new bundle goes to a **new dataset slug** whose licence field reads **CC-BY-NC-4.0**, not CC0; never version `rsna-kad-weights` v5 or touch kernel v10 | tester §2, dev §3 | developer | 09-27 (first new export) | 0.5 h |
| Release package: `LICENSE` (Apache-2.0 for code), model card, third-party licences (ConvNeXt-V2 NC, Gemma ToU, pilkwang/Claude provenance), label tables **stripped of `*_evidence`, jsonl, few-shots and gold**, pinned requirements x3, `docs/reproduce.md`, `export_kaggle.py --verify`, exclude `qa/competitor/kernels` and `.kaggle` | dev P2-10/11 | developer, reviewer signs off | draft 10-18, final 11-05 | 2 d |
| Prune configs/runs, unify `load_bundle`/`load_member` | dev P2-12 | developer | **after 10-22 only** | 0.5-1 d |

## 6. Risk register (top 8)

| # | Risk | Likelihood / impact | Mitigation |
|---|---|---|---|
| 1 | Private shake-up: the public split is ~390 studies, prevalence may differ, and the fork mass is LB-tuned | high / high | Equal weights, no LB tuning. Two main picks by default (§3). Decide on clean paired probes only. |
| 2 | Pseudo-labels confirm the teacher's errors, and contaminated metrics hide it | medium / high | Weak-OOF never decides. Pre-registered probe gates G1-G3. Gold veto. main_v3 remains the fallback pick. |
| 3 | **Claude-quota exhaustion stalls the project** (labels and agents share one plan) | high / high | The label pass is capped (≤ 2 pilot-equivalents/day) with a hard stop on 10-04. From 10-12 the quota is reserved for engineering and verification. Every GPU job is a self-running screen chain. A one-page runbook lets the user run export/verify/submit by hand. |
| 4 | **Idle GPUs across session ends** (2.5 days lost already) | high / medium | Each lane keeps a chain queued >= 24 h deep, with the next speculative job pre-queued behind a kill-switch file. A daily `nvidia-smi` + `screen -ls` check at session start. The calendar names a default job for every gap. |
| 5 | Kaggle runtime overrun (main_v3's real runtime is unknown; new members; the hang path) | medium / high | Read the runtime now. The §2.7 x1.25 rule. Replace members rather than add them. Timeout and fallback fixes by 10-06. The wall guard counts from process start. |
| 6 | Release error: a half-trained fold, a stale notebook, a dataset/kernel mismatch, a mis-selection in the UI | medium / high | Export guard (09-27). Git tags. New slugs. Kernel-vs-local verification for every submission. Select 1 day early with a screenshot. |
| 7 | A rules premise is wrong (two false scares already) or rules/host posts change | low-medium / high | Every premise is quoted from primary text in §4. Re-read the rules page and host posts on 10-08, 10-15 and 10-20 (efficiency eligibility, external-data/LLM use, licence). A disclosure of LLM label sources is ready for the write-up. |
| 8 | Hidden-test differences (compressed syntaxes, `Fluid_Sensitive != Fat_Suppression`, missing slots) | low / high | main_v3 already ran the hidden set. The tester's 13 drills and 4 transcodes pass. Keep the prior-row alarm. Add a real RLE/J2K round-trip test (B5). |

## 7. Proposed updates to docs/architecture.md (for the coordinator to apply)

- §2.1: replace "Hosted APIs excluded (rule 4.b)" with "allowed (host 733965, 08-09/08-27); pilkwang table allowed (user 09-25); Claude pass via the Max plan, quota-capped".
- §2.4/§2.6: record the real arms (A convnext_small@384 8 ep, E effv2s@256, C convnextv2_tiny@384 + 90 mm view) and add "equal weight per **arm**, asserted at export".
- §4.1 (B6): log main_v3 = submission 56488491, dataset v5, kernel v10, LB 0.933, suite 275, and the measured hidden runtimes once read.
- §4.2 addendum: pseudo-label and new-target runs are judged on consensus + gold-58 (near-clean, teacher site-leak caveat) + a pre-registered paired LB probe (>= +0.010 against a scored twin). Weak-OOF is printed, never decisive.
- §4.3: annotate "Stage-9 NOT BOUGHT" as judged on a blind arbiter; superseded by `pseudo_v3` (09-25).
- New §4.4 "Endgame (09-26 -> 10-22)": bets 1-4, the STOP list, gates G1 09-28 / G2 10-01 / G3 10-08, freeze 10-15, final ensemble 10-18, submissions 10-20, selection 10-21.
- §2.8/release: every bundle goes to a new dataset slug with licence CC-BY-NC-4.0; `rsna-kad-weights` v5 and kernel v10 are frozen.
- §5: add the efficiency-eligibility clause and the slot-2 decision rule; update the Kaggle-runtime line with the measured value; add risks 3 (quota) and 4 (idle GPUs).
