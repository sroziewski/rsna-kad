"""Knee side per study from three weak sources: the Laterality tag, the
SeriesDescription, and the FOV centre x in patient coordinates (LPS: +x = patient
left). The geometric sign is fitted per manufacturer on tagged studies and only
trusted where it agrees with the tags (architecture §2.3)."""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from rsna_kad.constants import ID_COL
from rsna_kad.dicom import Header, image_centre

TAG_CONF_CONSISTENT = 0.95
TAG_CONF_CONTRADICTORY = 0.6
DESC_CONF = 0.8
# Planes whose FOV centre is used, in order of preference (architecture §2.3).
GEOM_PLANES = ("Axial", "Coronal", "Sagittal")

_DESC_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(^|[\s_\-])(RT|R|RIGHT|RECHTS|DROIT|DER)([\s_\-]|$)", re.I), "R"),
    (re.compile(r"(^|[\s_\-])(LT|L|LEFT|LINKS|GAUCHE|IZQ)([\s_\-]|$)", re.I), "L"),
)


@dataclass(frozen=True)
class Side:
    side_tag: str
    side_desc: str
    side_geom: str
    side_est: str
    side_conf: float
    tag_conf: float = 0.0
    centre_x: float = float("nan")


@dataclass(frozen=True)
class GeomRule:
    sign: int        # +1: centre_x > 0 means left (plain LPS); -1: the opposite
    agreement: float
    n: int
    trusted: bool


_FAMILIES = (("siemens", "siemens"), ("philips", "philips"), ("gehc", "ge"), ("ge medical", "ge"),
             ("toshiba", "canon"), ("canon", "canon"), ("fujifilm", "hitachi"), ("hitachi", "hitachi"))


def manufacturer_family(name: str | None) -> str:
    """Vendor family: the 12 Manufacturer spellings share conventions within a vendor, and
    e.g. 'Philips Healthcare' alone has no tagged studies to fit a rule on."""
    low = (name or "").strip().lower()
    for needle, family in _FAMILIES:
        if needle in low:
            return family
    return low or "unknown"


def parse_laterality_tag(value: str | None) -> str:
    v = (value or "").strip().upper()
    if v in ("L", "LEFT"):
        return "L"
    if v in ("R", "RIGHT"):
        return "R"
    return ""


def parse_desc_side(desc: str | None) -> str:
    text = (desc or "").strip()
    if not text or text == "DummySeriesDesc!":
        return ""
    hits = {side for pat, side in _DESC_PATTERNS if pat.search(text)}
    return hits.pop() if len(hits) == 1 else ""


def majority(values: Sequence[str]) -> tuple[str, float]:
    """(winner, share of votes); ('', 0.0) when nothing votes."""
    votes = Counter(v for v in values if v)
    if not votes:
        return "", 0.0
    side, n = votes.most_common(1)[0]
    return side, n / sum(votes.values())


def series_centre_x(headers: Sequence[Header]) -> float:
    return float(np.mean([image_centre(h)[0] for h in headers]))


def study_centre_x(centre_by_plane: Mapping[str, Sequence[float]]) -> float:
    for plane in GEOM_PLANES:
        xs = [x for x in centre_by_plane.get(plane, ()) if np.isfinite(x)]
        if xs:
            return float(np.mean(xs))
    return float("nan")


def side_from_geometry(centre_x: float, rule: GeomRule | None) -> str:
    if rule is None or not np.isfinite(centre_x):
        return ""
    return "L" if centre_x * rule.sign > 0 else "R"


def combine(side_tag: str, tag_conf: float, side_desc: str, side_geom: str,
            rule: GeomRule | None) -> tuple[str, float]:
    if side_tag:
        return side_tag, tag_conf
    if side_desc:
        return side_desc, DESC_CONF
    if side_geom and rule is not None and rule.trusted:
        return side_geom, rule.agreement
    return "unknown", 0.0


def aggregate_series(rows: Sequence[Mapping]) -> tuple[str, float, str, float]:
    """Per-study tag/description vote from per-series rows holding side_tag, side_desc,
    plane and centre_x. Returns (side_tag, tag_conf, side_desc, centre_x)."""
    tag, share = majority([str(r.get("side_tag") or "") for r in rows])
    tag_conf = 0.0 if not tag else (TAG_CONF_CONSISTENT if share == 1.0 else TAG_CONF_CONTRADICTORY)
    desc, _ = majority([str(r.get("side_desc") or "") for r in rows])
    by_plane: dict[str, list[float]] = {}
    for r in rows:
        by_plane.setdefault(str(r.get("plane")), []).append(float(r.get("centre_x", np.nan)))
    return tag, tag_conf, desc, study_centre_x(by_plane)


def fit_geom_convention(study_df: pd.DataFrame, min_studies: int = 30,
                        min_agreement: float = 0.95) -> dict[str, GeomRule]:
    """Sign of centre_x ↔ side per manufacturer family, fitted on consistently tagged studies."""
    rules: dict[str, GeomRule] = {}
    cx = pd.to_numeric(study_df["centre_x"], errors="coerce")
    df = study_df[(study_df["side_tag"].isin(["L", "R"])) & (study_df["tag_conf"] >= TAG_CONF_CONSISTENT)
                  & np.isfinite(cx)]
    for manu, grp in df.groupby(df["manufacturer"].map(manufacturer_family)):
        left_positive = ((grp["centre_x"] > 0) == (grp["side_tag"] == "L")).mean()
        sign = 1 if left_positive >= 0.5 else -1
        agreement = float(max(left_positive, 1.0 - left_positive))
        n = int(len(grp))
        rules[str(manu)] = GeomRule(sign, agreement, n, n >= min_studies and agreement >= min_agreement)
    return rules


def estimate_sides(study_df: pd.DataFrame, rules: Mapping[str, GeomRule]) -> pd.DataFrame:
    """Add side_geom, side_est, side_conf to a per-study frame with side_tag, tag_conf,
    side_desc, centre_x, manufacturer."""
    out = study_df.copy()
    geom, est, conf = [], [], []
    for r in out.to_dict("records"):
        rule = rules.get(manufacturer_family(str(r["manufacturer"])))
        g = side_from_geometry(float(r["centre_x"]), rule)
        e, c = combine(str(r["side_tag"]), float(r["tag_conf"]), str(r["side_desc"]), g, rule)
        geom.append(g)
        est.append(e)
        conf.append(c)
    out["side_geom"], out["side_est"], out["side_conf"] = geom, est, np.asarray(conf, dtype=np.float32)
    return out


def estimate_side(headers_by_series: Mapping[str, Sequence[Header]], series_df: pd.DataFrame,
                  rules: Mapping[str, GeomRule] | None = None) -> Side:
    """Single-study path (inference): headers of every series of one study + its
    series_df rows (Anatomical_Plane)."""
    plane_of = dict(zip(series_df["SeriesInstanceUID"], series_df["Anatomical_Plane"]))
    rows = []
    manufacturer = ""
    for uid, headers in headers_by_series.items():
        if not headers:
            continue
        manufacturer = manufacturer or headers[0].manufacturer
        rows.append({
            "side_tag": parse_laterality_tag(headers[0].laterality),
            "side_desc": parse_desc_side(headers[0].series_description),
            "plane": plane_of.get(uid, ""),
            "centre_x": series_centre_x(headers),
        })
    tag, tag_conf, desc, cx = aggregate_series(rows)
    rule = (rules or {}).get(manufacturer_family(manufacturer))
    geom = side_from_geometry(cx, rule)
    est, conf = combine(tag, tag_conf, desc, geom, rule)
    return Side(tag, desc, geom, est, conf, tag_conf, cx)


def agreement_report(study_df: pd.DataFrame, rules: Mapping[str, GeomRule]) -> pd.DataFrame:
    """side_geom vs side_tag agreement per manufacturer (tagged studies only)."""
    rows = []
    for manu, rule in sorted(rules.items()):
        rows.append({"manufacturer": manu, "n_tagged": rule.n, "sign": rule.sign,
                     "agreement": round(rule.agreement, 4), "trusted": rule.trusted})
    rep = pd.DataFrame(rows, columns=["manufacturer", "n_tagged", "sign", "agreement", "trusted"])
    counts = study_df.groupby(study_df["manufacturer"].map(manufacturer_family))[ID_COL].count()
    return rep.merge(counts.rename("n_studies"), left_on="manufacturer", right_index=True, how="left")
