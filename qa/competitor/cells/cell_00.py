# =============================== RUN CONFIG =================================
# Every tunable weight in the pipeline, in one place. Defaults reproduce the
# 0.942 run exactly.
#
# Note how much of this differs from the 0.941 lineage: the A5 weight, the Rad
# alpha and the Rad second-pass alpha were all raised, and every Raptor arm now
# samples 94 windows instead of 62/42. Only the outer CoAtNet routing is
# unchanged. Those inner moves are as much of the gap as the extra model is.
import os

PRESET = os.environ.get("RSNA_PRESET", "speedy")

RUN = {
    # --- stage 2: A5 attention folds against the DINO base ------------------
    "a5_w": 0.52,                 # 0.941 lineage used 0.45

    # --- stage 3: RadImageNet ----------------------------------------------
    "rad_alpha": 0.55,            # 0.941 lineage used 0.50
    "rad_e13_member_w": 0.50,
    "rad_second_alpha": 0.20,     # 0.941 lineage used 0.15
    "rad_twin_alt_w": 0.500001,
    "rad_cal_w": 0.40,            # 88-feature calibrator, on gated findings only

    # --- stage 4: Raptor views ---------------------------------------------
    "raptor_view_w": {"maxspan-v5": 0.60, "native384dense-v10": 0.10,
                      "maxspan-v5-reverse": 0.10, "native384-v8": 0.20},
    # Windows sampled per study by every arm. The released arms carry 62 (42
    # for the short-slot v8 view); this run overrides all of them to 94, which
    # samples each stack more densely. It is the accuracy change that needed no
    # new model, and it is only affordable because of the speed work below:
    # shared DINO prefixes, persistent T4x2 replicas, cached decoding.
    "raptor_k_eval": 94,

    # CoAt family blended into the public Raptor arm. Two members when both
    # children succeed - the residual resgated top-3 and the D4 depth-zone SWA3
    # - combined by equal per-finding rank averaging, then mixed in at alpha.
    "coat_private_alpha": 0.40,

    # --- final fusion: CoAtNet arm against the transformer stack -----------
    # Identical to the 0.941 routing. Lateral Meniscus at 1.00 discards the
    # DINO, A5 and RadImageNet stages entirely for that column - three of four
    # stages, on the strength of a public split that is a fraction of the test
    # set. It may be real. It is also the likeliest place to give back points
    # privately, which is what the "parent" preset is for.
    "coatnet_w": {"__default__": 0.60, "ACL": 0.75, "Medial Meniscus": 0.80,
                  "Lateral Meniscus": 1.00, "Lateral OA": 0.75, "Fracture": 0.75},
}

if PRESET == "parent":
    RUN["coatnet_w"] = {"__default__": 0.60}
elif PRESET == "halfway":
    RUN["coatnet_w"] = {name: (0.60 + weight) / 2 if name != "__default__" else 0.60
                        for name, weight in RUN["coatnet_w"].items()}
elif PRESET == "sparse":
    # The window count back at its released value, for a runtime-constrained
    # run or to measure what the denser sampling actually bought.
    RUN["raptor_k_eval"] = None
elif PRESET != "speedy":
    raise RuntimeError(f"unknown RSNA_PRESET {PRESET!r}; use speedy, parent, "
                       f"halfway or sparse")


def coat_w(label):
    return float(RUN["coatnet_w"].get(label, RUN["coatnet_w"]["__default__"]))


print(f"[config] preset={PRESET} a5_w={RUN['a5_w']} rad_alpha={RUN['rad_alpha']} "
      f"k_eval={RUN['raptor_k_eval']}", flush=True)
print("[config] outer CoAtNet weight per finding:",
      {k: v for k, v in RUN["coatnet_w"].items() if k != "__default__"} or "flat 0.60",
      flush=True)