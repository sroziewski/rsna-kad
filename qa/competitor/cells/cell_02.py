# ==================== EDA 1 — what is inside a study =======================
# A study is a bag of series, and no two bags match. Everything downstream —
# the whole "slot" idea — exists to turn that variable bag into a fixed tensor.
if _EDA_TRAIN_SERIES is not None:
    frame = _EDA_TRAIN_SERIES
    per_study = frame.groupby("StudyInstanceUID").size()
    planes = frame["Anatomical_Plane"].value_counts()
    combos = (frame.assign(
        combo=frame["Anatomical_Plane"].astype(str) + " / " +
              _eda_np.where(frame.get("Fat_Suppression", 0) > 0, "fat-sat", "no fat-sat"))
        ["combo"].value_counts())

    figure, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    axes[0].hist(per_study, bins=range(int(per_study.min()), int(per_study.max()) + 2),
                 color=_EDA_INK)
    axes[0].set_title(f"Series per study (median {per_study.median():.0f})")
    axes[0].set_xlabel("series")
    axes[1].bar(planes.index.astype(str), planes.values, color=_EDA_INK)
    axes[1].set_title("Acquisition plane")
    axes[2].barh(combos.index[::-1], combos.values[::-1], color=_EDA_INK)
    axes[2].set_title("Plane x fat suppression")
    plt.tight_layout()
    plt.show()

    print("Coverage of the six slots the models expect, over training studies:")
    wanted = [("Sagittal", 1), ("Sagittal", 0), ("Coronal", 1),
              ("Coronal", 0), ("Axial", 1), ("Axial", 0)]
    rows = []
    for plane, fat in wanted:
        have = frame[(frame["Anatomical_Plane"] == plane) &
                     (frame.get("Fat_Suppression", 0) == fat)]
        rows.append({"slot": f"{plane} {'fat-sat' if fat else 'no fat-sat'}",
                     "studies with it": have["StudyInstanceUID"].nunique(),
                     "share": have["StudyInstanceUID"].nunique() /
                              frame["StudyInstanceUID"].nunique()})
    coverage = _eda_pd.DataFrame(rows)
    print(coverage.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nA slot no study fills is a column of zeros fed to the encoder; a study "
          "that fills none of them can only be given a constant, and under ROC-AUC "
          "a block of identical constants earns half credit against everything it "
          "ties with. That is the cheapest score leak in this whole pipeline.")