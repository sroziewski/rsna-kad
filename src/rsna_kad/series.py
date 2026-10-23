"""Series → slot assignment from the CSV flags (the only complete description of
series type, data_analysis.md §3) and the per-slot ranking/fallback rules."""
from __future__ import annotations

from typing import Mapping, Sequence

import pandas as pd

from rsna_kad.constants import ID_COL, SERIES_COL, SLOT_FALLBACK, SLOT_OF, SLOTS

RANK_COLS = ("is_3d", "n_slices", "series_number")


def slot_of(plane: str, fs: int | bool | str) -> str:
    key = (str(plane).strip().capitalize(), int(fs))
    if key not in SLOT_OF:
        raise ValueError(f"no slot for plane={plane!r} fs={fs!r}")
    return SLOT_OF[key]


def _rank_key(row: Mapping) -> tuple[int, int, int]:
    # 2D before 3D, then most slices, then lowest SeriesNumber (architecture §2.2).
    return (int(bool(row.get("is_3d", False))), -int(row.get("n_slices", 0) or 0),
            int(row.get("series_number", 0) or 0))


def select_for_slot(candidates: Sequence[Mapping]) -> str:
    """SeriesInstanceUID of the preferred candidate; each candidate is a mapping with
    SeriesInstanceUID and optionally is_3d / n_slices / series_number."""
    if not candidates:
        raise ValueError("select_for_slot: no candidates")
    best = min(candidates, key=lambda r: (_rank_key(r), str(r[SERIES_COL])))
    return str(best[SERIES_COL])


def assign_slots(series_df: pd.DataFrame, slots: Sequence[str] = SLOTS) -> pd.DataFrame:
    """One row per (study, slot, series): native rows (fallback=False) ranked within
    their slot, plus fallback rows (fallback=True) that lend the FS=0 plane's series
    to a missing FS slot. Columns: StudyInstanceUID, slot, SeriesInstanceUID, rank,
    selected, fallback, native_slot."""
    df = series_df.copy()
    df["native_slot"] = [slot_of(p, fs) for p, fs in zip(df["Anatomical_Plane"], df["Fluid_Sensitive"])]
    for col in RANK_COLS:
        if col not in df.columns:
            df[col] = 0
    if "ok" in df.columns:
        df = df[df["ok"].astype(bool)].copy()
    df["_key"] = [_rank_key(r) for r in df[list(RANK_COLS)].to_dict("records")]
    df = df.sort_values([ID_COL, "native_slot", "_key", SERIES_COL], kind="stable")
    df["rank"] = df.groupby([ID_COL, "native_slot"]).cumcount()
    df["slot"] = df["native_slot"]
    df["fallback"] = False
    native = df[[ID_COL, "slot", SERIES_COL, "rank", "fallback", "native_slot"]]

    present = native.groupby(ID_COL)["slot"].agg(set)
    extra = []
    for slot, source in SLOT_FALLBACK.items():
        if slot not in slots:
            continue
        missing = [s for s, have in present.items() if slot not in have and source in have]
        rows = native[(native[ID_COL].isin(missing)) & (native["slot"] == source)].copy()
        rows["slot"] = slot
        rows["fallback"] = True
        extra.append(rows)
    out = pd.concat([native, *extra], ignore_index=True) if extra else native.reset_index(drop=True)
    out = out[out["slot"].isin(list(slots))]
    out["selected"] = out["rank"] == 0
    cols = [ID_COL, "slot", SERIES_COL, "rank", "selected", "fallback", "native_slot"]
    return out[cols].sort_values([ID_COL, "slot", "rank"]).reset_index(drop=True)


def slot_table(assign_df: pd.DataFrame) -> dict[str, dict[str, list[str]]]:
    """{study: {slot: [series in rank order]}} including fallback rows."""
    table: dict[str, dict[str, list[str]]] = {}
    ordered = assign_df.sort_values([ID_COL, "slot", "rank"])
    for study, slot, series in zip(ordered[ID_COL], ordered["slot"], ordered[SERIES_COL]):
        table.setdefault(study, {}).setdefault(slot, []).append(series)
    return table
