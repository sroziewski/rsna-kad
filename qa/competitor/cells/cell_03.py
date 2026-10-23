# ================== EDA 2 — the findings, and how they co-occur ============
# Twelve binary findings, scored as twelve separate AUCs and averaged. That
# framing matters: a rare finding counts exactly as much as a common one, so
# the rare columns are where a submission is won or lost.
if _EDA_TRAIN is not None and set(_EDA_TARGETS).issubset(_EDA_TRAIN.columns):
    gold = _EDA_TRAIN.dropna(subset=_EDA_TARGETS)
    if len(gold):
        prevalence = gold[_EDA_TARGETS].mean().sort_values()
        figure, axes = plt.subplots(1, 2, figsize=(13, 4.2))
        axes[0].barh(prevalence.index, prevalence.values, color=_EDA_INK)
        axes[0].set_xlabel(f"prevalence among {len(gold)} fully-labelled studies")
        axes[0].set_title("How often each finding is positive")
        for y, (name, value) in enumerate(prevalence.items()):
            axes[0].text(value + 0.005, y, f"{value:.2f}", va="center", fontsize=8)

        matrix = gold[_EDA_TARGETS].astype(float).corr()
        image = axes[1].imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)
        axes[1].set_xticks(range(len(_EDA_TARGETS)))
        axes[1].set_xticklabels(_EDA_TARGETS, rotation=90, fontsize=7)
        axes[1].set_yticks(range(len(_EDA_TARGETS)))
        axes[1].set_yticklabels(_EDA_TARGETS, fontsize=7)
        axes[1].set_title("Co-occurrence between findings")
        axes[1].grid(False)
        figure.colorbar(image, ax=axes[1], shrink=0.8)
        plt.tight_layout()
        plt.show()

        pairs = (matrix.where(_eda_np.triu(_eda_np.ones(matrix.shape), 1).astype(bool))
                 .stack().sort_values(ascending=False))
        print("Most correlated pairs:")
        for (left, right), value in pairs.head(5).items():
            print(f"  {left:<18} + {right:<18} {value:+.2f}")
        print("\nThe three osteoarthritis compartments move together, which is why "
              "the calibrator downstream groups them. Findings that co-occur are "
              "also findings an ensemble tends to get right or wrong together — "
              "worth remembering before reading any per-finding weight as skill.")
    else:
        print("[eda] no fully-labelled studies found in train.csv", flush=True)