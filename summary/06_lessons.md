# 6. What worked, what didn't, and how we decided

## 6.1 What moved the score

| Lever | Gain | Who |
|---|---|---|
| LLM report labels instead of rules (Qwen3-32B, 2 prompts) | gold labels 0.816 → 0.852 | Szymon |
| Better labels plus a stronger backbone at 384 px (MVP → arm A; the MVP recipe on LLM labels already scored 0.913) | LB 0.865 → 0.931 | Szymon |
| Diverse label voters (public table / Gemma-3) | +0.012-0.015 label AUC | Szymon |
| **Pseudo-labels round 1** (0.5 label + 0.5 teacher OOF) | gold +0.0135, **LB +0.010** (0.933 → 0.943) | Szymon |
| Full-data retrains, C-only ensemble | LB 0.943 → 0.948 | Szymon |
| Public models (R7, Raptor) with our C leg added (plain R7 alone: 0.956) | LB 0.948 → 0.957 | Szymon |
| Team merge: rank-blending independent model families | LB 0.957 → 0.959 | Szymon + Qile + Andrey |
| **Test-time Gaussian blur** (0.288 mm R6/R7, 0.8 px hires) | R7 +0.0101 gold; LB 0.959 → 0.960/0.961 | Andrey |
| Out-of-fold stackers (stacker5, S36, S48) | LB 0.961; U8 the best 0.961 at full precision, per Andrey | Andrey |

## 6.2 Dead ends (tie or loss on the pre-declared gate)

- **Labels:** "strict" prompt, confident-learning relabelling (only 0.10 % of cells contradicted), Claude Sonnet
  voter, silentq, glmgrade (GLM grading of "mild"), agreement masking, Opus-label retrain, stale-teacher removal,
  t70 heavier teacher, multi-source teachers, pseudo round 2.
- **Models:**
  - CoAtNet arm B (slow, underfit), ConvNeXt-V2-base (tie at 3x the cost), per-label attention and transformer heads;
  - Pipeline B (DINOv3): diversity disappeared once both families were pseudo-labelled (ρ 0.77 → 0.90);
  - max-pool MIL, nnPU loss, DANN against hospital (**hurt**, −0.0126), pairwise loss, MRI intensity augmentation;
  - synovitis second-stage classifier.
- **Inference:** z-TTA (+0.0005), crop TTA (LB tie), rotation TTA (screened on gold; only crop went to LB), blur on our own C leg (no effect), learned stackers
  over our own 10 models (all lost; they only helped later on diverse team members, out of fold).
- **External data:** OAI-trained C (LB tie), MRNet / KneeMRI C (LB tie), GLM-5.3 vision labeller (0.66-0.73 gold).
- **Ensembles:** stk (re-teaching on the public stack lost diversity), r6base, P+A without V, equal thirds, s6r,
  bpblur, bpsyn, bprefit, the U-series, the C3SE expert: all ties at the noise level.
- **Merges declined:** candidates whose gold-58 predictions did not raise ours, or who offered no own models.

## 6.3 Decision rules (how they evolved)

1. **09-21 (`qa/arbiter`, AR §4.2):** accept at ≥ +0.010 with P ≥ 0.95. 0.005-0.010 needs every metric to agree.
   Below 0.005 is a tie. Gold-58 is a **veto**. The public LB (~390 studies) resolves only ~0.02, so it was never
   used for ranking.
2. **09-26 (AR §4.5):** paired differences only. Accept at ≥ +0.004 on gold with no drop below −0.009, plus one
   paired LB probe.
3. **10-03 on:** a pre-declared test on 3 full-data seeds against twins, paired bootstrap on gold-58: accept iff
   Δ ≥ +0.004 at P ≥ 0.95. Per-label deltas were reported but not acted on.
4. **10-15 / 10-16:** Andrey showed that gold under-rates models that help on normal knees. From then on, the judges for
   extra-knee legs treated gold as information only, with a floor of −0.004 (AR L1316-1317).
5. **10-20 on:** LB-vs-gold slope 0.24 and an LB SD of ~0.003, so a new pick needed **LB ≥ 0.962**. Variants were
   pre-registered with their own rules (e.g. C3SE: "finals only if above C3S in the UI order").

## 6.4 Lessons

- **Labels mattered more than architecture.** The biggest jumps came from report labels and pseudo-labels. More arms
  of the same kind saturated (+0.0077, +0.0049, +0.0007).
- **Diversity has to be measured.** It disappears once members are trained on the same teacher: Pipeline B,
  stk, and the Opus-label C all drifted towards the existing members.
- **Gold-58 is small and biased** (no normal knees, enriched for positives). Use it as a veto with paired bootstrap,
  never for weight tuning. Weights tuned on gold are circular.
- **The Kaggle time budget is a correctness issue.** A leg timed out silently in joint3 v1 (confirmed by diag
  code 3). joint3blurall's ERROR was attributed to the budget when joint3blurall2 passed with an 8.8 h budget (leg A unblurred,
  `--diag-encode`).
  The s6r ERROR was most likely a timeout too, but that is not proven (AR L1495-1503, L1587). `--diag-encode` (the failure code shows up in the LB score) was the only way to see it on
  the hidden test. `--require-all-legs` reports a timeout as a crash.
- **Pin the docker image.** Kaggle's switch to torch 2.11 broke our decoder wheels.
- **Pre-declare gates before seeing results.** Across solo and team work, this kept ~40 tie-level experiments from
  being adopted on noise.
- **Last-epoch selection.** Best-epoch selection is ~+0.008 optimistic.

## 6.5 Gaps left

- The LB scores of 57193974 (v4pf) and 57298855 (blend5) are not recorded.
- Fit details of the S48 stacker exist only in Andrey's repo (`research/d_stacker_s48_6.json`).
- Never confirmed: whether gold-58 was kept out of Qile's FULL retrains.
- About 20 qa/ directories exist only on the server (see [07_scripts_inventory.md](07_scripts_inventory.md)).
- Two infra bugs were never fixed: a timed-out leg's grandchild processes survive, and an R7 failure inside leg V
  does not show up in the diag codes.
