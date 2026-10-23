# ============================ EDA — setup ==================================
# Everything below is read-only and runs in a couple of minutes. It exists to
# answer three questions before any model runs:
#   what is actually in a study, how are the findings distributed, and what do
#   the reports look like — because the labels behind every weight here were
#   read out of those reports.
import warnings
from pathlib import Path as _EdaPath

import matplotlib.pyplot as plt
import numpy as _eda_np
import pandas as _eda_pd

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 9})
_EDA_INK = "#2f6f9f"
_EDA_WARM = "#c8622d"

_EDA_ROOT = next((p for p in (
    _EdaPath("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
    _EdaPath("/kaggle/input/rsna-knee-abnormality-detection"))
    if (p / "test.csv").is_file()), None)

_EDA_TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
                "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
                "Contusion", "Fracture"]

if _EDA_ROOT is None:
    print("[eda] competition data not attached; skipping", flush=True)
    _EDA_TRAIN = _EDA_TRAIN_SERIES = _EDA_TEST_SERIES = None
else:
    _EDA_TRAIN = _eda_pd.read_csv(_EDA_ROOT / "train.csv", dtype={"StudyInstanceUID": str})
    _EDA_TRAIN_SERIES = _eda_pd.read_csv(_EDA_ROOT / "train_series.csv",
                                         dtype={"StudyInstanceUID": str,
                                                "SeriesInstanceUID": str})
    _EDA_TEST_SERIES = _eda_pd.read_csv(_EDA_ROOT / "test_series.csv",
                                        dtype={"StudyInstanceUID": str,
                                               "SeriesInstanceUID": str})
    _eda_gold = _EDA_TRAIN.dropna(subset=[c for c in _EDA_TARGETS
                                          if c in _EDA_TRAIN.columns])
    print(f"[eda] {len(_EDA_TRAIN):,} training studies, "
          f"{len(_EDA_TRAIN_SERIES):,} training series, "
          f"{len(_EDA_TEST_SERIES):,} test series", flush=True)
    print(f"[eda] {len(_eda_gold):,} studies carry all twelve labels; the rest are "
          f"supervised only by what a text model read out of the report", flush=True)