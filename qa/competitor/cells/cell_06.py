# ============== EDA 6 — scanners, sites, and the leak in the folds =========
# MRI intensity is not calibrated. A scanner leaves a fingerprint, and a model
# given random folds will happily learn to recognise the scanner instead of the
# knee. That inflates any offline score and none of it survives a private split
# drawn from different sites.
import pydicom as _site_dicom

if _EDA_ROOT is not None:
    root = (_EDA_ROOT / "train_series" if (_EDA_ROOT / "train_series").is_dir()
            else _EDA_ROOT / "test_series")
    makers, models, fields, sampled = [], [], [], 0
    for study_dir in sorted(root.iterdir())[:250]:
        if not study_dir.is_dir():
            continue
        series_dirs = [d for d in sorted(study_dir.iterdir()) if d.is_dir()]
        if not series_dirs:
            continue
        files = sorted(series_dirs[0].glob("*.dcm"))
        if not files:
            continue
        try:
            header = _site_dicom.dcmread(str(files[len(files) // 2]),
                                         stop_before_pixels=True, force=True)
        except Exception:
            continue
        makers.append(str(getattr(header, "Manufacturer", "unknown")).strip() or "unknown")
        models.append(str(getattr(header, "ManufacturerModelName", "unknown")).strip()
                      or "unknown")
        try:
            fields.append(float(getattr(header, "MagneticFieldStrength", 0) or 0))
        except Exception:
            fields.append(0.0)
        sampled += 1

    if sampled:
        maker_counts = _eda_pd.Series(makers).value_counts()
        model_counts = _eda_pd.Series(models).value_counts().head(10)
        figure, axes = plt.subplots(1, 2, figsize=(13, 3.8))
        axes[0].barh(maker_counts.index[::-1], maker_counts.values[::-1], color=_EDA_INK)
        axes[0].set_title(f"Manufacturer over {sampled} sampled studies")
        axes[1].barh(model_counts.index[::-1], model_counts.values[::-1], color=_EDA_WARM)
        axes[1].set_title("Scanner model (top 10)")
        axes[1].tick_params(axis="y", labelsize=7)
        plt.tight_layout()
        plt.show()

        field_counts = _eda_pd.Series([f for f in fields if f > 0]).value_counts()
        print("Field strengths present:",
              ", ".join(f"{value:.1f}T x{count}" for value, count in field_counts.items()))
        top = model_counts.iloc[0] / sampled
        print(f"The single most common scanner model covers {top:.0%} of the sample.")
        print("\nThe training path in this notebook groups folds by md5(report text), "
              "which deduplicates identical reports but does nothing about the "
              "scanner. Manufacturer + model + matrix size + pixel spacing is a "
              "serviceable site key and costs nothing to compute from headers you "
              "are already reading. Grouping on it is what makes an offline number "
              "mean something.")