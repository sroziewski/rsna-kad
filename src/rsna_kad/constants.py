"""Names, orders and paths shared by every stage. Nothing here may change once
a cache, label table or checkpoint bundle has been produced with it."""
from __future__ import annotations

import os

# Column names of sample_submission.csv, in submission order.
LABELS: tuple[str, ...] = (
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
    "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
)
# Snake-case keys used in parquet columns, JSON schemas and code; same order.
LABEL_KEYS: tuple[str, ...] = (
    "acl", "mcl", "med_men", "lat_men", "med_oa", "lat_oa",
    "pf_oa", "effusion", "synovitis", "baker", "contusion", "fracture",
)
KEY_TO_LABEL: dict[str, str] = dict(zip(LABEL_KEYS, LABELS))
LABEL_TO_KEY: dict[str, str] = dict(zip(LABELS, LABEL_KEYS))

# Labels whose report mention rate is far below gold prevalence (research.md §1):
# "not mentioned" must not be treated as a confident negative for these.
UNDER_REPORTED_KEYS: frozenset[str] = frozenset(
    {"synovitis", "fracture", "contusion", "baker", "med_oa", "lat_oa", "pf_oa"})

STATES: tuple[str, ...] = ("affirmed", "hedged", "negated", "absent")
SEVERITIES: tuple[str, ...] = ("none", "mild", "moderate", "severe", "unspecified")

# Series slots: (Anatomical_Plane, Fluid_Sensitive). Fluid_Sensitive == Fat_Suppression
# in every row of the data (data_analysis.md §2).
SLOTS: tuple[str, ...] = ("sag_fs", "cor_fs", "ax_fs", "sag_t1", "cor_t1", "ax_t1")
SLOT_OF: dict[tuple[str, int], str] = {
    ("Sagittal", 1): "sag_fs", ("Coronal", 1): "cor_fs", ("Axial", 1): "ax_fs",
    ("Sagittal", 0): "sag_t1", ("Coronal", 0): "cor_t1", ("Axial", 0): "ax_t1",
}
SLOT_FALLBACK: dict[str, str] = {"sag_fs": "sag_t1", "cor_fs": "cor_t1", "ax_fs": "ax_t1"}
MAIN_SLOTS: tuple[str, ...] = ("sag_fs", "cor_fs", "ax_fs", "sag_t1")

ID_COL = "StudyInstanceUID"
SERIES_COL = "SeriesInstanceUID"

# Server locations; overridable so tests and the Kaggle notebook can point elsewhere.
DATA_DIR = os.environ.get("RSNA_KAD_DATA", "/raid/sroziewski/data/rsna_kad")
CACHE_DIR = os.environ.get("RSNA_KAD_CACHE", "/raid/sroziewski/cache/rsna_kad")
HF_HOME = os.environ.get("HF_HOME", "/raid/sroziewski/cache/huggingface")
