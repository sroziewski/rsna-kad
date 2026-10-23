# =============== EDA 7 — how much of each series the model actually sees ====
# The CoAtNet arm asks for a fixed number of slices per slot (18 sagittal
# fat-sat, 14 sagittal, 12 coronal fat-sat, 8 coronal, 12 axial) and then
# samples 62 three-slice windows from the 64-slice stack. Whether that is
# generous or lossy depends on how long the series actually are.
if _EDA_TRAIN_SERIES is not None and "n_slices" in dir():
    pass

if _EDA_ROOT is not None:
    root = (_EDA_ROOT / "test_series" if (_EDA_ROOT / "test_series").is_dir()
            else _EDA_ROOT / "train_series")
    lengths = []
    for study_dir in sorted(root.iterdir())[:200]:
        if not study_dir.is_dir():
            continue
        for series_dir in sorted(study_dir.iterdir()):
            if series_dir.is_dir():
                lengths.append((series_dir.parent.name, series_dir.name,
                                len(list(series_dir.glob("*.dcm")))))
    if lengths:
        counts = _eda_np.array([n for _, _, n in lengths])
        asks = {"Sagittal fat-sat": 18, "Sagittal": 14, "Coronal fat-sat": 12,
                "Coronal": 8, "Axial": 12}
        figure, axes = plt.subplots(1, 2, figsize=(13, 3.6))
        axes[0].hist(counts[counts < _eda_np.percentile(counts, 99)], bins=50,
                     color=_EDA_INK)
        for name, ask in asks.items():
            axes[0].axvline(ask, color=_EDA_WARM, linestyle="--", linewidth=0.8)
        axes[0].set_title("Slices per series (dashed = what a slot asks for)")
        axes[0].set_xlabel("slices")

        shares = [(counts >= ask).mean() for ask in asks.values()]
        axes[1].bar(list(asks), shares, color=_EDA_INK)
        axes[1].set_ylim(0, 1)
        axes[1].set_title("Share of series long enough to fill their slot")
        axes[1].tick_params(axis="x", rotation=20)
        plt.tight_layout()
        plt.show()

        print(f"Median series length {_eda_np.median(counts):.0f} slices; "
              f"{(counts < 12).mean():.1%} are shorter than twelve.")
        print("Where a series is shorter than its slot, the sampler repeats "
              "slices, so several of the 62 windows are duplicates and the "
              "attention pooling sees the same evidence more than once. Where a "
              "series is much longer, the sampler is skipping anatomy — and a "
              "meniscus tear visible on two slices can fall between samples "
              "entirely. That is the mechanism behind the max and top-2 window "
              "pooling used for the focal findings upstream.")