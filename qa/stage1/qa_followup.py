import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "/raid/sroziewski/sandbox/rsna_kad/src")
from rsna_kad import laterality, cv

CACHE = Path("/raid/sroziewski/cache/rsna_kad/v1")
REPO = Path("/raid/sroziewski/sandbox/rsna_kad")
df = pd.read_parquet(CACHE / "index.parquet")
tr = df[df["split"] == "train"]

# (a) 2D rows where n_cached != min(n_src, 40)
d2 = df[~df["is_3d"]]
mis = d2[d2["n_cached"] != np.minimum(d2["n_src"], 40)]
print("2D n_cached mismatch rows:", len(mis))
print(mis[["SeriesInstanceUID", "manufacturer", "n_src", "n_cached", "z_step_mm", "z_step_cached_mm", "slot"]].head(10).to_string())
print("n_src>40 & 2D:", int((d2["n_src"] > 40).sum()), " of which n_cached==40:", int(((d2["n_src"] > 40) & (d2["n_cached"] == 40)).sum()))

# (b) study-level laterality agreement with the developer's aggregation (axial > coronal > sagittal)
rows = []
for study, g in tr.groupby("StudyInstanceUID"):
    by_plane = {}
    for p, x in zip(g["plane"], g["centre_x"]):
        by_plane.setdefault(p, []).append(float(x))
    cx = laterality.study_centre_x(by_plane)
    r = g.iloc[0]
    rows.append({"study": study, "fam": laterality.manufacturer_family(r["manufacturer"]), "cx": cx,
                 "side_tag": r["side_tag"], "tag_conf": float(r["tag_conf"]), "side_geom": r["side_geom"],
                 "side_est": r["side_est"], "side_conf": float(r["side_conf"]), "side_desc": r["side_desc"],
                 "n_manu": g["manufacturer"].nunique()})
st = pd.DataFrame(rows)
print("studies:", len(st), "multi-manufacturer studies:", int((st["n_manu"] > 1).sum()))
tagged = st[st["side_tag"].isin(["L", "R"])]
out = {}
for fam, g in tagged.groupby("fam"):
    cons = g[g["tag_conf"].round(2) == 0.95]
    def agree(h):
        lp = ((h["cx"] > 0) == (h["side_tag"] == "L")).mean()
        return round(float(max(lp, 1 - lp)), 4), (1 if lp >= 0.5 else -1)
    a_all, s_all = agree(g)
    a_con, s_con = agree(cons)
    gv = g[g["side_geom"] != ""]
    out[fam] = dict(n_tagged=len(g), n_consistent=len(cons), agree_all=a_all, agree_consistent=a_con, sign=s_con,
                    side_geom_eq_tag_all=round(float((gv["side_geom"] == gv["side_tag"]).mean()), 4) if len(gv) else None,
                    n_contradictory=len(g) - len(cons))
print(json.dumps(out, indent=1))
print("contradictory studies:", int((st["tag_conf"].round(2) == 0.6).sum()),
      "side_conf==0.6:", int((st["side_conf"].round(2) == 0.6).sum()),
      "their families:", st[st["tag_conf"].round(2) == 0.6]["fam"].value_counts().to_dict())
# Siemens/Philips untagged studies: does side_geom agree with side_desc where both exist?
u = st[(st["side_tag"] == "") & (st["side_desc"] != "") & (st["side_geom"] != "")]
print("untagged with desc+geom:", len(u), "geom==desc:", round(float((u["side_geom"] == u["side_desc"]).mean()), 4) if len(u) else None,
      u.groupby("fam").apply(lambda h: round(float((h["side_geom"] == h["side_desc"]).mean()), 3)).to_dict())
# side_est source counts
def src(r):
    if r["side_tag"]: return "tag"
    if r["side_desc"]: return "desc"
    if r["side_est"] in ("L", "R"): return "geom"
    return "none"
st["src"] = st.apply(src, axis=1)
print("side_est source:", st["src"].value_counts().to_dict())
print(pd.crosstab(st["fam"], st["src"]).to_string())
print("side_conf by source:", st.groupby("src")["side_conf"].agg(["min", "max"]).round(4).to_dict("index"))
# |cx| distribution: is the centre clearly off-midline?
print("abs(cx) quantiles by fam:", st.groupby("fam")["cx"].apply(lambda s: np.round(np.percentile(np.abs(s), [5, 50, 95]), 1).tolist()).to_dict())

# (c) folds: language crosstab per fold (counts)
fl = cv.load_folds(REPO / "data/folds/folds_v1.csv")
ct = pd.crosstab(fl["fold"], fl["lang"])
print(ct.to_string())
print((pd.crosstab(fl["fold"], fl["lang"], normalize="index") * 100).round(1).to_string())
# how many sites per language, largest site share within language
ss = fl.groupby("lang").agg(n=("StudyInstanceUID", "size"), n_sites=("site_key", "nunique"))
ss["largest_site_share"] = fl.groupby("lang").apply(lambda h: round(h["site_key"].value_counts(normalize=True).iloc[0], 3))
print(ss.to_string())
# language shares with the sklearn method for comparison
fs = cv.make_folds(REPO / "analysis/out/study_summary.csv", 5, 2026, method="sklearn")
print("sklearn sizes:", fs["fold"].value_counts(normalize=True).sort_index().round(3).to_dict(),
      "gold:", fs.groupby("fold")["is_gold"].sum().to_dict())
d = (pd.crosstab(fs["fold"], fs["lang"], normalize="index") - fs["lang"].value_counts(normalize=True)).abs()
print("sklearn max lang dev pp:", round(float(d.max().max()) * 100, 2))
