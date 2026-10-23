You are reviewing a Kaggle medical-imaging solution. The attached file `pipelineA_core.py` is a
discussion excerpt of our pipeline (the full code exists and is verified; this file keeps only the
ideas). Read its docstring carefully: it lists our results and everything we already tried that
did not help.

## Context you need

- Task: knee MRI study -> 12 binary findings, metric macro ROC AUC. 4,407 training studies; only
  58 have image-level ("gold") labels, the rest only a multilingual free-text radiology report.
  Test set = images only.
- Our best: public LB 0.948 (leader ~0.961). Public LB noise is about +-0.002-0.003.
- Validation: the 58 gold studies, never trained on, paired bootstrap. We accept a change only if
  it gains >= +0.004 on gold-58 with P >= 0.95 over 3 seeds.
- Weakest findings on gold-58 (ensemble AUC): synovitis 0.81, lateral OA 0.84, PF OA 0.88,
  lateral meniscus 0.89-0.90. Strongest: ACL, effusion, Baker's, medial OA ~0.97-0.98.
- Key measured facts:
  - The pseudo-label step (0.5 x report label + 0.5 x teacher prediction) gave our largest gain
    (+0.010 LB). After it, every model we train converges to the same targets: an independent
    DINOv3 ViT-B pipeline correlated 0.77 with ours before pseudo-labelling (blend +0.012) and
    0.90 after (blend -0.0004). Models with different losses on the same targets correlate 0.97.
  - Eight model-side changes in a row tied or lost: stacking, pseudo-label round 2, pairwise
    ranking loss, MRI intensity augmentation, crop/rotation TTA, an independent pipeline merge,
    a max-pool residual next to attention pooling, and non-negative PU loss for under-reported
    findings (nnPU lowered synovitis 0.81 -> 0.78).
  - Laterality canonicalisation (flipping left knees) changed AUC by < 0.001.
  - More averaging does not help: a 12-seed ensemble scored LB 0.947, and 6 seeds + 3 max-pool +
    3 nnPU models scored 0.948 (gold-58 unchanged in both).
- Already proposed by another reviewer and rejected without testing (re-propose only with a
  materially new argument):
  - any use of the 58 gold studies for training or meta-learning (linear-probe fine-tuning on
    gold, bilevel L2RW sample reweighting with gold as the meta-set, a nearest-neighbour memory
    bank of gold embeddings) - 58 studies, some findings with only a handful of positives, and
    they are our only clean measurement;
  - per-finding query attention ("one query per finding" MIL) - equivalent to our per-label
    attention heads, tried;
  - hard-coded anatomical attention priors (e.g. "synovitis query looks at the upper 40 % of
    axial slices") and a separate medial-vs-lateral compartment head;
  - adaptive / denser slice sampling - training already samples up to 36 of <= 40 slices per
    view and inference covers every slice;
  - ensembles of view-specialist models (each model sees a different subset of views) - not
    tested; we expect the shared targets, not the inputs, to make them converge too.
  - Report labels come from a local LLM (Qwen3-32B) + rules + a second LLM + a public label table;
    a further LLM voter added only +0.004 on a 481-report pilot.
- Constraints: 2x A6000 GPUs for ~18 days; one full training run takes ~2 h; Kaggle inference on
  2x T4 within 9 h; 5 submissions/day; the 58 gold studies must stay out of training (they are
  our only clean measurement); pretrained weights of any licence are allowed.

## What I want

Propose AT MOST 5 ideas that are genuinely NEW relative to the docstring and the list above.
Focus where gains are still plausible:
1. the label / target side (how reports and the teacher become training targets, how to break
   the convergence of all models onto the same pseudo-targets);
2. the weak findings (synovitis, lateral OA, PF OA, lateral meniscus);
3. anything that adds real diversity to an ensemble without retraining on the same targets.

For each idea give:
- **Idea** in 2-3 sentences, concrete enough to implement.
- **Why it should work HERE** - tied to a specific fact above, not a generic best practice.
- **Expected gold-58 effect** with an honest range (including "probably small").
- **Cheapest decisive test** within <= 3 training runs (~6 GPU-hours), and what result would
  reject it.
- **Main failure risk** and whether it touches the 58 gold studies (it must not train on them).

Rank them by expected gain per GPU-hour. Do NOT write code. Do NOT re-propose anything in the
"tried" list unless you name what is materially different and why that difference matters.
Do not invent numbers about our pipeline that are not stated here or in the file; if you assume
something, say so.
