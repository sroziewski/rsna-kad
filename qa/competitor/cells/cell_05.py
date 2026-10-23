# ============= EDA 4 — geometry, and why the crop is in millimetres ========
# Pixel spacing varies across scanners, so a fixed pixel crop would show a
# different amount of knee per study. Every stage here crops in millimetres
# instead. This is what that variation looks like.
import pydicom as _eda_dicom

if _EDA_ROOT is not None:
    series_root = (_EDA_ROOT / "test_series" if (_EDA_ROOT / "test_series").is_dir()
                   else _EDA_ROOT / "train_series")
    spacings, rows_cols, sampled = [], [], 0
    for study_dir in sorted(series_root.iterdir())[:120]:
        if not study_dir.is_dir():
            continue
        for series_dir in sorted(study_dir.iterdir()):
            files = sorted(series_dir.glob("*.dcm"))
            if not files:
                continue
            try:
                header = _eda_dicom.dcmread(str(files[len(files) // 2]),
                                            stop_before_pixels=True, force=True)
                spacings.append(float(header.PixelSpacing[0]))
                rows_cols.append((int(header.Rows), int(header.Columns)))
                sampled += 1
            except Exception:
                continue
        if sampled > 400:
            break

    if spacings:
        spacings = _eda_np.array(spacings)
        figure, axes = plt.subplots(1, 2, figsize=(13, 3.6))
        axes[0].hist(spacings, bins=50, color=_EDA_INK)
        axes[0].set_title(f"In-plane pixel spacing over {sampled} series")
        axes[0].set_xlabel("mm per pixel")
        sizes = _eda_pd.Series([f"{r}x{c}" for r, c in rows_cols]).value_counts().head(8)
        axes[1].barh(sizes.index[::-1], sizes.values[::-1], color=_EDA_INK)
        axes[1].set_title("Most common matrix sizes")
        plt.tight_layout()
        plt.show()

        crop_mm = 140.0
        pixels = crop_mm / spacings
        print(f"A {crop_mm:.0f} mm crop spans {pixels.min():.0f} to {pixels.max():.0f} "
              f"pixels across these series — a {pixels.max() / pixels.min():.1f}x range. "
              f"Cropping a fixed number of pixels instead would hand the model a "
              f"different anatomical field of view per scanner, which is the kind of "
              f"artefact a model will happily learn to read as signal.")