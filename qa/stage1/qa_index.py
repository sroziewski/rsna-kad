"""Stage-1 QA: cache index contract, laterality recomputation, folds contract.
Read-only on the cache; writes only qa_index_out.json next to itself."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/raid/sroziewski/sandbox/rsna_kad/src")
from rsna_kad import laterality, series as series_mod, cv  # noqa: E402

CACHE = Path("/raid/sroziewski/cache/rsna_kad/v1")
REPO = Path("/raid/sroziewski/sandbox/rsna_kad")
OUT = {}

df = pd.read_parquet(CACHE / "index.parquet")
OUT["n_rows"] = len(df)
OUT["per_split"] = df["split"].value_counts().to_dict()
OUT["n_ok"] = int(df["ok"].sum())
OUT["errors"] = df.loc[~df["ok"], ["StudyInstanceUID", "SeriesInstanceUID", "error"]].to_dict("records")

CONTRACT = ["split", "StudyInstanceUID", "SeriesInstanceUID", "plane", "fs", "slot", "rank", "selected",
            "is_3d", "n_src", "n_cached", "spacing_mm", "fov_mm", "z_step_mm", "crop_mm", "px", "p_lo",
            "p_hi", "truncated_slices", "side_tag", "side_desc", "side_geom", "side_est", "side_conf",
            "ok", "error"]
OUT["missing_cols"] = [c for c in CONTRACT if c not in df.columns]
OUT["extra_cols"] = [c for c in df.columns if c not in CONTRACT]
OUT["dtypes"] = {c: str(df[c].dtype) for c in df.columns}
OUT["n_cached_min_max"] = [int(df["n_cached"].min()), int(df["n_cached"].max())]
OUT["n_cached_gt40"] = int((df["n_cached"] > 40).sum())
OUT["n_cached_lt1"] = int((df["n_cached"] < 1).sum())
OUT["px_unique"] = df["px"].unique().tolist()
OUT["crop_mm_unique"] = df["crop_mm"].unique().tolist()
OUT["dup_series"] = int(df.duplicated(["split", "SeriesInstanceUID"]).sum())
OUT["n_studies"] = int(df["StudyInstanceUID"].nunique())

# cross-check against train_series.csv
ts = pd.read_csv("/raid/sroziewski/data/rsna_kad/train_series.csv", dtype=str)
tr = df[df["split"] == "train"]
OUT["train_csv_rows"] = len(ts)
OUT["train_series_set_equal"] = set(ts["SeriesInstanceUID"]) == set(tr["SeriesInstanceUID"])
m = tr.merge(ts, on=["StudyInstanceUID", "SeriesInstanceUID"])
OUT["plane_fs_match_csv"] = bool((m["plane"] == m["Anatomical_Plane"]).all()
                                  and (m["fs"].astype(int) == m["Fluid_Sensitive"].astype(int)).all())
OUT["plane_iop_vs_csv_mismatch"] = int((df["plane_iop"] != df["plane"]).sum())

# sample 300 npy files: dtype, shape, n_cached match, mean
rng = np.random.default_rng(20260918)
samp = df[df["ok"]].sample(300, random_state=20260918)
bad = []
means = []
for r in samp.itertuples(index=False):
    p = CACHE / r.split / r.StudyInstanceUID / f"{r.SeriesInstanceUID}.npy"
    a = np.load(p, mmap_mode="r")
    if a.dtype != np.uint8 or a.ndim != 3 or a.shape[1:] != (384, 384) or not (1 <= a.shape[0] <= 40) \
            or a.shape[0] != r.n_cached:
        bad.append((str(p), str(a.dtype), a.shape, int(r.n_cached)))
    means.append(float(np.asarray(a).mean()))
OUT["sample300_bad"] = bad
OUT["sample300_mean_in_20_200"] = float(np.mean([(20 <= x <= 200) for x in means]))
OUT["sample300_mean_vs_index_maxabs"] = float(np.max(np.abs(np.asarray(means) - samp["mean_u8"].to_numpy())))
# sidecar presence
OUT["sample300_sidecar_missing"] = int(sum(
    not (CACHE / r.split / r.StudyInstanceUID / f"{r.SeriesInstanceUID}.json").exists() for r in samp.itertuples()))

# total size (du) + sum of expected sizes
du = subprocess.run(["du", "-sb", str(CACHE)], capture_output=True, text=True).stdout.split()[0]
OUT["du_bytes"] = int(du)
OUT["du_gib"] = round(int(du) / 2**30, 2)
OUT["expected_npy_bytes_from_index"] = int((df["n_cached"] * 384 * 384 + 128).sum())
OUT["n_npy_files"] = int(subprocess.run(
    f"find {CACHE} -name '*.npy' | wc -l", shell=True, capture_output=True, text=True).stdout.strip())
OUT["n_tmp_files"] = int(subprocess.run(
    f"find {CACHE} -name '*.tmp*' | wc -l", shell=True, capture_output=True, text=True).stdout.strip())
OUT["manifest"] = json.loads((CACHE / "manifest.json").read_text())

# truncated files
pm = pd.read_csv(REPO / "analysis/out/pixel_size_mismatch.csv")
trunc_series = [p.split("/")[2] for p in pm["path"]]
OUT["truncated_expected_series"] = trunc_series
OUT["truncated_rows"] = df[df["SeriesInstanceUID"].isin(trunc_series)][
    ["SeriesInstanceUID", "n_src", "n_cached", "truncated_slices", "ok"]].to_dict("records")
OUT["truncated_total"] = int(df["truncated_slices"].sum())
OUT["truncated_other_series"] = int(((df["truncated_slices"] > 0) & ~df["SeriesInstanceUID"].isin(trunc_series)).sum())

# intensity
OUT["mean_u8_in_range_frac"] = float(df["mean_u8"].between(20, 200).mean())
OUT["mean_u8_out_of_range"] = df.loc[~df["mean_u8"].between(20, 200),
                                     ["SeriesInstanceUID", "manufacturer", "slot", "mean_u8", "p_lo", "p_hi",
                                      "fov_mm"]].to_dict("records")
OUT["mean_u8_describe"] = df["mean_u8"].describe().round(2).to_dict()
OUT["p_lo_describe"] = df["p_lo"].describe().round(2).to_dict()
OUT["p_hi_describe"] = df["p_hi"].describe().round(2).to_dict()
OUT["p_hi_le_p_lo"] = int((df["p_hi"] <= df["p_lo"]).sum())
OUT["p_lo_eq_0"] = int((df["p_lo"] == 0).sum())
OUT["p_lo_lt_0"] = int((df["p_lo"] < 0).sum())
OUT["p_lo_by_manu"] = df.groupby(df["manufacturer"].map(laterality.manufacturer_family))[["p_lo", "p_hi"]] \
    .median().round(1).to_dict("index")
OUT["p_lo_eq0_by_manu"] = df[df["p_lo"] == 0].groupby(
    df["manufacturer"].map(laterality.manufacturer_family)).size().to_dict()
OUT["fov_lt_130_n"] = int((df["fov_mm"] < 130).sum())
OUT["fov_describe"] = df["fov_mm"].describe().round(1).to_dict()
OUT["p_lo_eq0_and_fov_lt130"] = int(((df["p_lo"] == 0) & (df["fov_mm"] < 130)).sum())

# slots
OUT["slot_series_counts"] = tr["slot"].value_counts().to_dict()
OUT["slot_studies_with_ge1"] = tr.groupby("slot")["StudyInstanceUID"].nunique().to_dict()
OUT["selected_per_study_slot_max"] = int(tr[tr["selected"]].groupby(["StudyInstanceUID", "slot"]).size().max())
OUT["studies_missing_selected_native"] = {
    s: int(tr["StudyInstanceUID"].nunique() - tr[(tr["slot"] == s) & tr["selected"]]["StudyInstanceUID"].nunique())
    for s in ["sag_fs", "cor_fs", "ax_fs", "sag_t1", "cor_t1", "ax_t1"]}
assigned = series_mod.assign_slots(tr.rename(columns={"n_src": "n_slices"}).assign(
    Anatomical_Plane=tr["plane"], Fluid_Sensitive=tr["fs"]))
fb = assigned[assigned["fallback"] & assigned["selected"]]
OUT["fallback_studies"] = fb.groupby("slot")["StudyInstanceUID"].nunique().to_dict()
studies_no_sagfs = set(tr["StudyInstanceUID"]) - set(tr[tr["slot"] == "sag_fs"]["StudyInstanceUID"])
studies_no_corfs = set(tr["StudyInstanceUID"]) - set(tr[tr["slot"] == "cor_fs"]["StudyInstanceUID"])
OUT["studies_without_sag_fs"] = len(studies_no_sagfs)
OUT["studies_without_cor_fs"] = len(studies_no_corfs)
OUT["studies_without_ax_fs"] = len(set(tr["StudyInstanceUID"]) - set(tr[tr["slot"] == "ax_fs"]["StudyInstanceUID"]))
# does the rank-0 selection prefer 2D over 3D, most slices, lowest series number?
sel = tr[tr["selected"]]
OUT["selected_3d_while_2d_available"] = int(sum(
    bool(g[g["selected"]]["is_3d"].any() and (~g["is_3d"]).any())
    for _, g in tr.groupby(["StudyInstanceUID", "slot"])))

# 3D
d3 = df[df["is_3d"]]
OUT["n_3d"] = len(d3)
OUT["n_3d_by_slot"] = d3["slot"].value_counts().to_dict()
OUT["n_3d_ncached_max"] = int(d3["n_cached"].max())
OUT["n_3d_nsrc_min_max"] = [int(d3["n_src"].min()), int(d3["n_src"].max())]
OUT["n_3d_zstep_src_describe"] = d3["z_step_mm"].describe().round(2).to_dict()
OUT["n_3d_zstep_cached_describe"] = d3["z_step_cached_mm"].describe().round(2).to_dict()
OUT["n_2d_nsrc_gt40"] = int(((~df["is_3d"]) & (df["n_src"] > 40)).sum())
OUT["n_2d_ncached_eq_min_nsrc_40"] = bool(((~df["is_3d"]) & (df["n_cached"] == np.minimum(df["n_src"], 40))).all())
OUT["n_3d_ncached_eq_40_frac"] = float((d3["n_cached"] == 40).mean())
OUT["is_3d_but_mracq_absent_gt60"] = int(((df["n_src"] > 60) & ~df["is_3d"]).sum())

# ------------------------------------------------------------- laterality
st = df[df["split"] == "train"].drop_duplicates("StudyInstanceUID").copy()
st["fam"] = st["manufacturer"].map(laterality.manufacturer_family)
OUT["side_tag_counts"] = st["side_tag"].value_counts().to_dict()
OUT["side_est_counts"] = st["side_est"].value_counts().to_dict()
OUT["side_conf_counts"] = st["side_conf"].round(4).value_counts().to_dict()
OUT["n_contradictory_tag_conf06"] = int((st["tag_conf"].round(3) == 0.6).sum())
OUT["contradictory_side_conf"] = st.loc[st["tag_conf"].round(3) == 0.6, "side_conf"].round(3).value_counts().to_dict()
# recompute contradictory studies from per-series tags (index side_tag is per-study after _finalize,
# so recompute from sidecars is expensive; instead count studies whose series tags disagree in the
# raw headers via tag_conf). Also recompute contradiction from the per-series 'side_tag' if present.
# source of side_est
def source(r):
    if r["side_tag"]:
        return "tag"
    if r["side_desc"]:
        return "desc"
    if r["side_geom"] and r["side_est"] in ("L", "R"):
        return "geom"
    return "none"
st["src"] = st.apply(source, axis=1)
OUT["side_est_source"] = st["src"].value_counts().to_dict()
OUT["side_est_source_by_fam"] = pd.crosstab(st["fam"], st["src"]).to_dict("index")
OUT["side_geom_nonempty_by_fam"] = st[st["side_geom"] != ""].groupby("fam").size().to_dict()
OUT["side_est_from_geom_by_fam"] = st[st["src"] == "geom"].groupby("fam").size().to_dict()
# agreement side_geom (sign of centre_x) vs side_tag on tagged studies, per family, recomputed
tagged = st[st["side_tag"].isin(["L", "R"]) & np.isfinite(st["centre_x"])]
agr = {}
for fam, g in tagged.groupby("fam"):
    left_pos = ((g["centre_x"] > 0) == (g["side_tag"] == "L")).mean()
    cons = g[g["tag_conf"].round(3) == 0.95]
    left_pos_c = ((cons["centre_x"] > 0) == (cons["side_tag"] == "L")).mean() if len(cons) else float("nan")
    agr[fam] = {"n_tagged": len(g), "n_consistent": len(cons), "agree_all": round(float(max(left_pos, 1 - left_pos)), 4),
                "agree_consistent": round(float(max(left_pos_c, 1 - left_pos_c)), 4),
                "sign": 1 if left_pos_c >= 0.5 else -1,
                "geom_vs_tag_on_tagged": round(float((g["side_geom"] == g["side_tag"]).mean()), 4) if (g["side_geom"] != "").any() else None}
OUT["geom_agreement_recomputed"] = agr
OUT["manifest_geom_rules"] = OUT["manifest"]["geom_rules"]
OUT["studies_by_fam"] = st["fam"].value_counts().to_dict()
OUT["centre_x_nan"] = int((~np.isfinite(st["centre_x"])).sum())
# untagged, undescribed, geometry not trusted -> unknown
OUT["side_desc_counts"] = st["side_desc"].value_counts().to_dict()
OUT["tag_vs_desc_conflict"] = int(((st["side_tag"] != "") & (st["side_desc"] != "") & (st["side_tag"] != st["side_desc"])).sum())
# test split
tst = df[df["split"] == "test"].drop_duplicates("StudyInstanceUID")
OUT["test_sides"] = tst[["StudyInstanceUID", "manufacturer", "side_tag", "side_desc", "side_geom", "side_est", "side_conf"]].to_dict("records")
OUT["test_rows"] = df[df["split"] == "test"][["SeriesInstanceUID", "slot", "rank", "selected", "is_3d", "n_src", "n_cached", "rows", "cols", "fov_mm", "mean_u8"]].to_dict("records")

# ------------------------------------------------------------- folds
folds = pd.read_csv(REPO / "data/folds/folds_v1.csv", dtype=str, keep_default_na=False)
OUT["folds_cols"] = list(folds.columns)
OUT["folds_rows"] = len(folds)
OUT["folds_md5"] = cv.md5_of(REPO / "data/folds/folds_v1.csv")
OUT["folds_unique_ids"] = bool(folds["StudyInstanceUID"].is_unique)
OUT["folds_values"] = sorted(folds["fold"].unique().tolist())
OUT["folds_is_gold_values"] = folds["is_gold"].value_counts().to_dict()
fl = cv.load_folds(REPO / "data/folds/folds_v1.csv")
OUT["folds_gold_total"] = int(fl["is_gold"].sum())
OUT["folds_sites_multi"] = int((fl.groupby("site_key")["fold"].nunique() > 1).sum())
OUT["folds_n_sites"] = int(fl["site_key"].nunique())
OUT["folds_sizes"] = fl["fold"].value_counts(normalize=True).sort_index().round(4).to_dict()
OUT["folds_gold_per_fold"] = fl.groupby("fold")["is_gold"].sum().to_dict()
glob = fl["lang"].value_counts(normalize=True)
per = pd.crosstab(fl["fold"], fl["lang"], normalize="index")
dev = (per - glob).abs()
OUT["folds_lang_global"] = glob.round(4).to_dict()
OUT["folds_lang_max_dev_pp"] = round(float(dev.max().max()) * 100, 2)
OUT["folds_lang_dev_pp_by_fold"] = (dev.max(axis=1) * 100).round(2).to_dict()
OUT["folds_lang_dev_pp_by_lang"] = (dev.max(axis=0) * 100).round(2).to_dict()
OUT["folds_check"] = cv.check_folds(fl)
# compare with study_summary (site_key, lang, labelled)
ss = pd.read_csv(REPO / "analysis/out/study_summary.csv", dtype=str, keep_default_na=False)
OUT["study_summary_cols"] = list(ss.columns)
idc = "StudyInstanceUID" if "StudyInstanceUID" in ss.columns else "study_dir"
mm = fl.merge(ss, left_on="StudyInstanceUID", right_on=idc, suffixes=("", "_ss"))
OUT["folds_vs_summary_rows"] = len(mm)
OUT["folds_vs_summary_site_match"] = bool((mm["site_key"] == mm["site_key_ss"]).all()) if "site_key_ss" in mm else None
OUT["folds_vs_summary_lang_match"] = bool((mm["lang"] == mm["lang_ss"]).all()) if "lang_ss" in mm else None
goldc = "labelled" if "labelled" in ss.columns else None
if goldc:
    OUT["folds_vs_summary_gold_match"] = bool((mm["is_gold"] == mm[goldc].str.lower().isin(["true", "1"])).all())
# folds ids vs train.csv
trcsv = pd.read_csv("/raid/sroziewski/data/rsna_kad/train.csv", dtype=str)
OUT["folds_ids_equal_train_csv"] = set(fl["StudyInstanceUID"]) == set(trcsv["StudyInstanceUID"])
OUT["folds_ids_equal_cache_train_studies"] = set(fl["StudyInstanceUID"]) == set(tr["StudyInstanceUID"])
# md5 pinned in test_cv.py
txt = (REPO / "tests/test_cv.py").read_text()
import re
OUT["folds_md5_pinned"] = re.search(r'FOLDS_V1_MD5 = "([0-9a-f]+)"', txt).group(1)
# regenerate with the config and compare
from rsna_kad.config import load_folds_config
fc = load_folds_config(REPO / "configs/folds_v1.yaml")
regen = cv.make_folds(REPO / fc.study_summary, fc.n_splits, fc.seed, fc.group_col, fc.strat_col, fc.method, fc.n_trials)
OUT["folds_regen_identical"] = bool(regen.sort_values("StudyInstanceUID").reset_index(drop=True)
                                    .equals(fl.sort_values("StudyInstanceUID").reset_index(drop=True)))

def conv(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, tuple):
        return list(o)
    return str(o)

Path(__file__).with_name("qa_index_out.json").write_text(json.dumps(OUT, indent=1, default=conv))
print(json.dumps(OUT, indent=1, default=conv))
