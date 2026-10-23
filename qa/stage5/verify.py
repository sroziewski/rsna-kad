"""Tester verification of stages 2+5 (labels + targets). CPU only, read-only on data."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path("/raid/sroziewski/sandbox/rsna_kad")
sys.path.insert(0, str(ROOT / "src"))
from rsna_kad.constants import LABEL_KEYS, LABELS, LABEL_TO_KEY, SEVERITIES, STATES, UNDER_REPORTED_KEYS  # noqa: E402
from rsna_kad.labels.schema import LABEL_TABLE_COLUMNS  # noqa: E402
from rsna_kad.labels.merge import MERGED_COLUMNS  # noqa: E402
from rsna_kad.labels.targets import TargetsConfig, cell_of, state_score  # noqa: E402
from rsna_kad.metrics import binarise, macro_auc  # noqa: E402

TRAIN = Path("/raid/sroziewski/data/rsna_kad/train.csv")
D = ROOT / "data"
OUT = ROOT / "qa" / "stage5"
MILD_KEYS = {"med_men", "lat_men", "effusion", "mcl", "med_oa", "lat_oa", "pf_oa"}
FINDINGS: list[str] = []


def sec(t):
    print(f"\n===== {t}")


train = pd.read_csv(TRAIN)
gold = train[train[LABELS[0]].notna()].rename(columns=LABEL_TO_KEY)[["StudyInstanceUID"] + list(LABEL_KEYS)].reset_index(drop=True)
gold_ids = set(gold.StudyInstanceUID)
folds = pd.read_csv(D / "folds" / "folds_v1.csv")
reports = dict(zip(train.StudyInstanceUID, train.Report.fillna("").astype(str)))
print("train rows", len(train), "gold", len(gold), "folds rows", len(folds), "folds is_gold", int(folds.is_gold.sum()))
print("gold ids == folds.is_gold ids:", gold_ids == set(folds.loc[folds.is_gold, "StudyInstanceUID"]))
print("gold_ids.txt == gold:", set((D / "labels" / "gold_ids.txt").read_text().split()) == gold_ids)

# ---------------------------------------------------------------- 2a/2b gold leakage
sec("2a few-shot / draft ids vs gold")
fs_ids, fs_reports = {}, {}
for f in sorted((D / "fewshot").glob("??.json")):
    ex = json.loads(f.read_text(encoding="utf-8"))
    fs_ids[f.stem] = [e["StudyInstanceUID"] for e in ex]
    fs_reports[f.stem] = [e["report"] for e in ex]
    print(f.stem, len(ex), "gold overlap:", len(set(fs_ids[f.stem]) & gold_ids))
draft = [l.strip() for l in (D / "fewshot" / "draft_ids.txt").read_text().splitlines() if l.strip()]
all_fs = {i for v in fs_ids.values() for i in v}
print("few-shot ids total", len(all_fs), "in train", len(all_fs & set(train.StudyInstanceUID)), "gold overlap", len(all_fs & gold_ids))
print("draft ids", len(draft), "gold overlap", len(set(draft) & gold_ids), "fewshot subset of draft:", all_fs <= set(draft))
sec("2b few-shot report text vs gold report text")
gold_texts = {reports[i].strip(): i for i in gold_ids}
n_same = 0
for lang, reps in fs_reports.items():
    for sid, r in zip(fs_ids[lang], reps):
        if r.strip() in gold_texts:
            n_same += 1
            print("  few-shot", lang, sid[-8:], "has text IDENTICAL to gold row", gold_texts[r.strip()][-8:])
        elif r.strip() != reports[sid].strip():
            print("  few-shot", lang, sid[-8:], "report text differs from train.csv text")
print("few-shot reports whose text equals a gold row's text:", n_same)
# the tr normal template: identical reports across many studies incl. gold?
tr_ids = set(train.StudyInstanceUID) & set(folds.loc[folds.lang == "tr", "StudyInstanceUID"])
cnt = Counter(reports[i].strip() for i in tr_ids)
tmpl_text, tmpl_n = cnt.most_common(1)[0]
tmpl_ids = sorted(i for i in tr_ids if reports[i].strip() == tmpl_text)
print("tr most common report text: n =", tmpl_n, "gold among them:", len(set(tmpl_ids) & gold_ids), "few-shot among them:", len(set(tmpl_ids) & all_fs))

# ---------------------------------------------------------------- 3 label-table contracts
sec("3 label table contracts")
tables = {}
for name in ("rules_v0", "llm_v1_raw", "llm_v2_raw", "llm_v1"):
    t = pd.read_parquet(D / "labels" / f"{name}.parquet")
    tables[name] = t
    missing = [c for c in LABEL_TABLE_COLUMNS if c not in t.columns]
    extra = [c for c in t.columns if c not in LABEL_TABLE_COLUMNS]
    exp_extra = set(MERGED_COLUMNS) - set(LABEL_TABLE_COLUMNS)
    bad_state = {k: sorted(set(t[f"{k}_state"].astype(str)) - set(STATES)) for k in LABEL_KEYS if set(t[f"{k}_state"].astype(str)) - set(STATES)}
    bad_sev = {k: sorted(set(t[f"{k}_severity"].astype(str)) - set(SEVERITIES)) for k in LABEL_KEYS if set(t[f"{k}_severity"].astype(str)) - set(SEVERITIES)}
    nan_state = int(sum(t[f"{k}_state"].isna().sum() for k in LABEL_KEYS))
    dts = {str(t[f"{k}_state"].dtype) for k in LABEL_KEYS} | {str(t[f"{k}_severity"].dtype) for k in LABEL_KEYS}
    ev_cells = ok_cells = 0
    for k in LABEL_KEYS:
        m = t[f"{k}_state"].astype(str) != "absent"
        ev_cells += int(m.sum())
        if f"{k}_evidence_ok" in t.columns:
            ok_cells += int(t.loc[m, f"{k}_evidence_ok"].sum())
    print(f"{name}: rows {len(t)} unique ids {t.StudyInstanceUID.nunique()} ids==train {set(t.StudyInstanceUID) == set(train.StudyInstanceUID)} "
          f"missing {missing} n_extra {len(extra)} extra_not_in_merged_spec {sorted(set(extra) - exp_extra)[:6]} "
          f"merged_spec_missing {sorted(exp_extra - set(t.columns))[:6] if name == 'llm_v1' else '-'} "
          f"bad_state {bad_state} bad_sev {bad_sev} nan_states {nan_state} dtypes {dts} "
          f"parse_ok {int(t.parse_ok.sum())}/{len(t)} evidence_ok {ok_cells}/{ev_cells} = {ok_cells / max(ev_cells, 1):.4f} "
          f"postop {int(t.postoperative.sum())} normal_template {int(t.normal_template.sum())} knee_side {t.knee_side.astype(str).value_counts().to_dict()}")
    # evidence on affirmed/negated only (the stage-5 acceptance wording)
    an = ok_an = 0
    for k in LABEL_KEYS:
        m = t[f"{k}_state"].astype(str).isin(["affirmed", "negated"])
        an += int(m.sum())
        if f"{k}_evidence_ok" in t.columns:
            ok_an += int(t.loc[m, f"{k}_evidence_ok"].sum())
    if an and f"acl_evidence_ok" in t.columns:
        print(f"   evidence_ok on affirmed/negated cells only: {ok_an}/{an} = {ok_an / an:.4f}")
    # severity/state consistency
    inc = sum(int(((t[f"{k}_state"].astype(str).isin(["negated", "absent"])) & (t[f"{k}_severity"].astype(str) != "none")).sum()) for k in LABEL_KEYS)
    inc2 = sum(int(((t[f"{k}_state"].astype(str).isin(["affirmed", "hedged"])) & (t[f"{k}_severity"].astype(str) == "none")).sum()) for k in LABEL_KEYS)
    inc3 = sum(int(((t[f"{k}_state"].astype(str) == "absent") & (t[f"{k}_evidence"].astype(str) != "")).sum()) for k in LABEL_KEYS)
    print(f"   neg/absent with severity!=none: {inc}; aff/hedged with severity none: {inc2}; absent with evidence: {inc3}")
for name in ("llm_v1_raw", "llm_v2_raw"):
    st = json.loads((D / "labels" / f"{name}.stats.json").read_text())
    print(name, "stats:", {k: st.get(k) for k in ("n_total", "n_resumed", "n_run", "parse_rate", "n_rows", "truncated", "evidence_ok_rate", "wall_s")})
    fp = D / "labels" / f"{name}.stats.fullpass.json"
    if fp.exists():
        st = json.loads(fp.read_text())
        print(name, "fullpass:", {k: st.get(k) for k in ("n_total", "n_resumed", "n_run", "parse_rate", "truncated", "evidence_ok_rate", "wall_s")})
    # jsonl integrity
    lines = (D / "labels" / f"{name}.jsonl").read_text(encoding="utf-8").split("\n")
    recs, torn = [], 0
    for l in lines:
        if not l.strip():
            continue
        try:
            recs.append(json.loads(l))
        except ValueError:
            torn += 1
    n_u2028 = sum(1 for l in lines if " " in l or " " in l or "\x85" in l)
    print(f"   jsonl torn lines {torn}; lines containing U+2028/2029/85 {n_u2028}")
    ids = [r["StudyInstanceUID"] for r in recs]
    fin = Counter(r.get("finish") for r in recs)
    print(f"   jsonl: {len(recs)} records, unique {len(set(ids))}, ==train {set(ids) == set(train.StudyInstanceUID)}, finish {dict(fin)}, parse_ok {sum(bool(r.get('parse_ok')) for r in recs)}, variant {Counter(r.get('variant') for r in recs)}")
    # parse_ok in jsonl vs table
    t = tables[name].set_index("StudyInstanceUID")
    mism = sum(bool(r.get("parse_ok")) != bool(t.loc[r["StudyInstanceUID"], "parse_ok"]) for r in recs)
    print(f"   parse_ok jsonl vs table mismatches: {mism}; failed ids: {[r['StudyInstanceUID'][-10:] for r in recs if not r.get('parse_ok')]}")
rules, v1, v2, merged = tables["rules_v0"], tables["llm_v1_raw"], tables["llm_v2_raw"], tables["llm_v1"]
print("rules has _mentioned cols:", all(f"{k}_mentioned" in rules.columns for k in LABEL_KEYS))
# merged consistency with raw: state == v1 state when v1 parsed
m = merged.set_index("StudyInstanceUID")
a = v1.set_index("StudyInstanceUID").loc[m.index]
b = v2.set_index("StudyInstanceUID").loc[m.index]
ok1, ok2 = a.parse_ok.to_numpy(bool), b.parse_ok.to_numpy(bool)
d1 = sum(int((m[f"{k}_state"].astype(str).to_numpy() != a[f"{k}_state"].astype(str).to_numpy())[ok1].sum()) for k in LABEL_KEYS)
d2 = sum(int((m[f"{k}_state_v2"].astype(str).to_numpy() != b[f"{k}_state"].astype(str).to_numpy())[ok1].sum()) for k in LABEL_KEYS)
print(f"merged primary != v1 (v1 parsed) cells: {d1}; merged _v2 != v2 cells: {d2}; merged parse_ok {int(merged.parse_ok.sum())}; v1 fail {int((~ok1).sum())} v2 fail {int((~ok2).sum())}")
v2_fail_ids = list(m.index[~ok2])
print("v2-failed ids:", [i[-10:] for i in v2_fail_ids], "their merged agree counts:", [int(sum(bool(m.loc[i, f'{k}_agree']) for k in LABEL_KEYS)) for i in v2_fail_ids],
      "their _v2 states:", [Counter(str(m.loc[i, f"{k}_state_v2"]) for k in LABEL_KEYS) for i in v2_fail_ids])

# ---------------------------------------------------------------- 4 gold-58 AUC
sec("4 gold-58 macro AUC recomputed with sklearn")


def scores(t: pd.DataFrame, avg_v2: bool) -> pd.DataFrame:
    out = pd.DataFrame({"StudyInstanceUID": t.StudyInstanceUID.to_numpy()})
    for k in LABEL_KEYS:
        s = np.array([state_score(a, b, k) for a, b in zip(t[f"{k}_state"].astype(str), t[f"{k}_severity"].astype(str))])
        if avg_v2 and f"{k}_state_v2" in t.columns:
            s2 = np.array([state_score(a, b, k) for a, b in zip(t[f"{k}_state_v2"].astype(str), t[f"{k}_severity_v2"].astype(str))])
            s = (s + s2) / 2
        out[k] = s
    return out


def my_score(state, sev, key):
    s = {"affirmed": 1.0, "hedged": 0.66, "absent": 0.33, "negated": 0.0}[state]
    if state == "affirmed" and sev == "mild" and key in MILD_KEYS:
        s -= 0.35
    return s


# independent implementation check of state_score
for st in STATES:
    for sv in SEVERITIES:
        for k in LABEL_KEYS:
            assert abs(my_score(st, sv, k) - state_score(st, sv, k)) < 1e-12
print("state_score == independent mapping for all state x severity x key")


def gold_auc(sc: pd.DataFrame, n_boot=1000, seed=0):
    df = gold.merge(sc, on="StudyInstanceUID", suffixes=("", "_p"))
    assert len(df) == 58
    y = df[list(LABEL_KEYS)].to_numpy(float)
    p = df[[f"{k}_p" for k in LABEL_KEYS]].to_numpy(float)
    per = {k: (roc_auc_score(y[:, j], p[:, j]) if 0 < y[:, j].sum() < len(y) else np.nan) for j, k in enumerate(LABEL_KEYS)}
    macro = float(np.nanmean(list(per.values())))
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        vals = [roc_auc_score(y[idx, j], p[idx, j]) for j in range(12) if 0 < y[idx, j].sum() < len(idx)]
        bs.append(np.mean(vals))
    ci = (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))) if bs else (np.nan, np.nan)
    return macro, per, ci, df


res = {}
for name, t, avg in (("rules_v0", rules, False), ("llm_v1_raw", v1, False), ("llm_v2_raw", v2, False),
                     ("llm_v1 merged (primary only)", merged, False), ("llm_v1 merged (V1/V2 avg = developer)", merged, True)):
    macro, per, ci, df = gold_auc(scores(t, avg))
    res[name] = (macro, per, ci)
    print(f"{name}: macro {macro:.4f} CI95 [{ci[0]:.3f}, {ci[1]:.3f}]  per label " + " ".join(f"{k}={v:.3f}" for k, v in per.items()))
for name in ("rules_v0", "llm_v1_raw", "llm_v2_raw", "llm_v1"):
    rep = json.loads((D / "labels" / f"{name}.gold.json").read_text())
    rep = rep["merged"] if "merged" in rep else rep
    print(f"  reported {name}: macro {rep['macro_auc']:.4f} ci {rep['ci95']} n {rep['n']} per_lang {{ {', '.join(f'{l}: n={d['n']} {d['macro_auc']:.3f}' for l, d in rep.get('per_lang', {}).items())} }}")
# per-language (merged, avg) with sklearn
_, _, _, df = gold_auc(scores(merged, True), n_boot=0)
df = df.merge(folds[["StudyInstanceUID", "lang"]], on="StudyInstanceUID")
for lang, sub in df.groupby("lang"):
    y = sub[list(LABEL_KEYS)].to_numpy(float)
    p = sub[[f"{k}_p" for k in LABEL_KEYS]].to_numpy(float)
    vals = [roc_auc_score(y[:, j], p[:, j]) for j in range(12) if 0 < y[:, j].sum() < len(y)]
    print(f"  merged per lang {lang}: n={len(sub)} macro {np.mean(vals):.3f} over {len(vals)} labels")
print("gold prevalence:", {k: round(float(gold[k].mean()), 3) for k in LABEL_KEYS})

# ---------------------------------------------------------------- 2c/3 targets
sec("2c + 3 targets v1")
cfg = TargetsConfig.load(ROOT / "configs" / "targets_v1.yaml")
print("cfg under_reported", sorted(cfg.under_reported), "== constants", set(cfg.under_reported) == UNDER_REPORTED_KEYS, "weights", cfg.weights)
fold_of = dict(zip(folds.StudyInstanceUID, folds.fold))
mf = merged.set_index("StudyInstanceUID")
calibs = {}
for k in range(5):
    cal = json.loads((D / "targets" / "v1" / f"calibration_fold{k}.json").read_text())
    calibs[k] = cal
    exp_n = sum(1 for g in gold_ids if fold_of[g] != k)
    n_gold_fold = sum(1 for g in gold_ids if fold_of[g] == k)
    # recompute counts from merged table + gold of training folds
    cnt_ok = True
    for key in LABEL_KEYS:
        c = Counter()
        for g in gold_ids:
            if fold_of[g] == k:
                continue
            cell = cell_of(str(mf.loc[g, f"{key}_state"]), str(mf.loc[g, f"{key}_severity"]))
            c[(cell, "n")] += 1
            c[(cell, "pos")] += int(gold.set_index("StudyInstanceUID").loc[g, key] > 0.5)
        for cell, (n, npos) in cal["counts"][key].items():
            if c[(cell, "n")] != n or c[(cell, "pos")] != npos:
                cnt_ok = False
    # recompute values: EB then PAVA
    val_ok = True
    from rsna_kad.labels.targets import pava_decreasing, state_prior_table, MONOTONE_CHAIN
    base = {(r.key, r.cell): r.base for r in state_prior_table(cfg).itertuples()}
    for key in LABEL_KEYS:
        vals = {c: (8 * base[(key, c)] + cal["counts"][key][c][1]) / (8 + cal["counts"][key][c][0]) for c in cal["counts"][key]}
        chain = pava_decreasing([vals[c] for c in MONOTONE_CHAIN], [8 + cal["counts"][key][c][0] for c in MONOTONE_CHAIN])
        for c, v in zip(MONOTONE_CHAIN, chain):
            if abs(v - cal["values"][key][c]) > 1e-9:
                val_ok = False
    mono = all(cal["values"][key]["affirmed"] >= cal["values"][key]["hedged"] - 1e-9 >= cal["values"][key]["absent"] - 2e-9 >= cal["values"][key]["negated"] - 3e-9 for key in LABEL_KEYS)
    mono2 = all(cal["values"][key]["affirmed"] >= cal["values"][key]["hedged"] and cal["values"][key]["hedged"] >= cal["values"][key]["absent"] and cal["values"][key]["absent"] >= cal["values"][key]["negated"] for key in LABEL_KEYS)
    strict_ties = sum(1 for key in LABEL_KEYS for a_, b_ in zip(MONOTONE_CHAIN, MONOTONE_CHAIN[1:]) if abs(cal["values"][key][a_] - cal["values"][key][b_]) < 1e-9)
    t = pd.read_parquet(D / "targets" / "v1" / f"fold{k}.parquet")
    cols = [f"{key}_{f}" for key in LABEL_KEYS for f in ("target", "weight")]
    vals_np = t[cols].to_numpy(np.float64)
    print(f"fold {k}: n_gold_train json {cal['n_gold_train']} recomputed {exp_n} (gold in fold {n_gold_fold}); counts match {cnt_ok}; EB+PAVA values match {val_ok}; "
          f"monotone {mono2} (ties {strict_ties}); targets rows {len(t)} unique {t.StudyInstanceUID.nunique()} gold rows {len(set(t.StudyInstanceUID) & gold_ids)} "
          f"cols {len(t.columns)} dtypes {set(str(d) for d in t[cols].dtypes)} nan {int(np.isnan(vals_np).sum())} min {vals_np.min():.3f} max {vals_np.max():.3f} "
          f"ids==train-gold {set(t.StudyInstanceUID) == set(train.StudyInstanceUID) - gold_ids}")
    print("   absent target per label: " + " ".join(f"{key}={cal['values'][key]['absent']:.3f}" for key in LABEL_KEYS))
    print("   hedged: " + " ".join(f"{key}={cal['values'][key]['hedged']:.3f}" for key in LABEL_KEYS))
    print("   affirmed/aff_mild/negated: " + " ".join(f"{key}={cal['values'][key]['affirmed']:.2f}/{cal['values'][key]['affirmed_mild']:.2f}/{cal['values'][key]['negated']:.3f}" for key in LABEL_KEYS))

# recompute targets+weights for all rows of fold 0 (and report on 200-row sample too)
sec("3 recompute targets/weights of fold0 from merged table (all rows; 200-row sample subset)")
t0 = pd.read_parquet(D / "targets" / "v1" / "fold0.parquet").set_index("StudyInstanceUID")
cal = calibs[0]
mm = mf.loc[t0.index]
w_of = {"affirmed": 1.0, "affirmed_mild": 1.0, "negated": 1.0, "hedged": 0.7}
n_bad_t = n_bad_w = 0
n_dis = n_hall = n_row = 0
bad_examples = []
for key in LABEL_KEYS:
    c1 = [cell_of(a, b) for a, b in zip(mm[f"{key}_state"].astype(str), mm[f"{key}_severity"].astype(str))]
    c2 = [cell_of(a, b) for a, b in zip(mm[f"{key}_state_v2"].astype(str), mm[f"{key}_severity_v2"].astype(str))]
    ev_ok = mm[f"{key}_evidence_ok"].to_numpy(bool)
    for i, (sid, ca, cb) in enumerate(zip(mm.index, c1, c2)):
        tgt = cal["values"][key][ca]
        w = w_of.get(ca, 0.1 if key in UNDER_REPORTED_KEYS else 0.5)
        if ca != cb:
            tgt = (tgt + cal["values"][key][cb]) / 2
            w *= 0.5
            n_dis += 1
        if not ev_ok[i]:
            w = min(w, 0.3)
            n_hall += 1
        if not bool(mm.loc[sid, "parse_ok"]):
            w = 0.2
        elif bool(mm.loc[sid, "postoperative"]):
            w = 0.2
            n_row += 1
        if abs(float(t0.loc[sid, f"{key}_target"]) - tgt) > 1e-6:
            n_bad_t += 1
            bad_examples.append((sid[-8:], key, ca, cb, tgt, float(t0.loc[sid, f"{key}_target"])))
        if abs(float(t0.loc[sid, f"{key}_weight"]) - w) > 1e-6:
            n_bad_w += 1
            bad_examples.append((sid[-8:], key, ca, cb, w, float(t0.loc[sid, f"{key}_weight"])))
print(f"fold0 all {len(t0)} rows x 12: target mismatches {n_bad_t}, weight mismatches {n_bad_w}; disagreement cells {n_dis}, hallucinated cells {n_hall}, postop cells {n_row}; examples {bad_examples[:5]}")
rng = np.random.default_rng(2026)
sample = rng.choice(t0.index.to_numpy(), 200, replace=False)
print("200-row sample ids checked (subset of the above):", len(sample), "weight distribution:", Counter(np.round(t0.loc[sample, [f'{k}_weight' for k in LABEL_KEYS]].to_numpy().ravel(), 3).tolist()))
wd = Counter(np.round(t0[[f"{k}_weight" for k in LABEL_KEYS]].to_numpy().ravel(), 3).tolist())
print("fold0 weight value distribution (all cells):", dict(sorted(wd.items())))
# targets v1 vs v0 difference
t0v0 = pd.read_parquet(D / "targets" / "v0" / "fold0.parquet").set_index("StudyInstanceUID")
common = t0.index.intersection(t0v0.index)
diff = (np.abs(t0.loc[common, [f"{k}_target" for k in LABEL_KEYS]].to_numpy() - t0v0.loc[common, [f"{k}_target" for k in LABEL_KEYS]].to_numpy()) > 1e-6).mean()
print(f"targets v1 vs v0 fold0: {len(common)} common rows, cells differing {diff:.3f}")

# ---------------------------------------------------------------- 6 consistency
sec("6 consistency")
for key in LABEL_KEYS:
    s1 = a[f"{key}_state"].astype(str).to_numpy()
    s2 = b[f"{key}_state"].astype(str).to_numpy()
    sr = rules.set_index("StudyInstanceUID").loc[m.index, f"{key}_state"].astype(str).to_numpy()
    both_ok = ok1 & ok2
    ag12 = float((s1 == s2)[both_ok].mean())
    # rules have no hedged: compare on affirmed/negated/absent collapse
    s1c = np.where(s1 == "hedged", "affirmed", s1)
    ag1r = float((s1c == sr).mean())
    ag1r_mentioned = float((s1c == sr)[(s1 != "absent") & (sr != "absent")].mean()) if ((s1 != "absent") & (sr != "absent")).any() else np.nan
    cell_dis = float((np.array([cell_of(x, y) for x, y in zip(s1, a[f"{key}_severity"].astype(str))]) != np.array([cell_of(x, y) for x, y in zip(s2, b[f"{key}_severity"].astype(str))])).mean())
    print(f"{key:10s} V1==V2 state {ag12:.3f} (cell-level disagreement {cell_dis:.3f})  V1==rules {ag1r:.3f} (both mentioned {ag1r_mentioned:.3f})  "
          f"mention_conflict {float(m[f'{key}_mention_conflict'].mean()):.3f}  V1 dist {dict(Counter(s1))}")
# spot-check 20 disagreeing cells -> targets
sec("6 disagreement -> targets spot-check (20 cells, fold0)")
dis_cells = [(sid, key) for key in LABEL_KEYS for sid in t0.index if not bool(mf.loc[sid, f"{key}_agree"]) and bool(mf.loc[sid, "parse_ok"])]
rng = np.random.default_rng(7)
for sid, key in [dis_cells[i] for i in rng.choice(len(dis_cells), 20, replace=False)]:
    r = mf.loc[sid]
    c1, c2 = cell_of(str(r[f"{key}_state"]), str(r[f"{key}_severity"])), cell_of(str(r[f"{key}_state_v2"]), str(r[f"{key}_severity_v2"]))
    exp_t = (cal["values"][key][c1] + cal["values"][key][c2]) / 2 if c1 != c2 else cal["values"][key][c1]
    print(f"  {sid[-8:]} {key:9s} v1={c1:13s} v2={c2:13s} rules={str(r[f'{key}_rules_state']):8s} ev_ok={bool(r[f'{key}_evidence_ok'])!s:5s} "
          f"target {float(t0.loc[sid, f'{key}_target']):.3f} (exp {exp_t:.3f}) weight {float(t0.loc[sid, f'{key}_weight']):.2f}")
# Turkish template rows
sec("6 Turkish normal-template rows")
print("template n =", tmpl_n, "gold:", len(set(tmpl_ids) & gold_ids))
for name, t in tables.items():
    tt = t.set_index("StudyInstanceUID").loc[tmpl_ids]
    sig = Counter(tuple(str(tt.loc[i, f"{k}_state"]) for k in LABEL_KEYS) for i in tmpl_ids)
    top = sig.most_common(1)[0]
    print(f"  {name}: distinct state signatures {len(sig)}; most common ({top[1]}/{tmpl_n}): negated={top[0].count('negated')} absent={top[0].count('absent')} affirmed={top[0].count('affirmed')} hedged={top[0].count('hedged')}; normal_template flags {int(tt.normal_template.sum())}")
    if len(sig) > 1:
        for s, n in sig.most_common()[1:4]:
            print("     other:", n, {k: v for k, v in zip(LABEL_KEYS, s) if v != top[0][LABEL_KEYS.index(k)]})
tt = t0.loc[[i for i in tmpl_ids if i in t0.index]]
print("  template rows in fold0 targets:", len(tt), "distinct target rows:", len(set(map(tuple, np.round(tt[[f'{k}_target' for k in LABEL_KEYS]].to_numpy(), 4).tolist()))))

# ---------------------------------------------------------------- 7 under-reporting
sec("7 affirmed rates vs gold prevalence")
for key in LABEL_KEYS:
    st = merged[f"{key}_state"].astype(str)
    print(f"{key:10s} affirmed {float((st == 'affirmed').mean()):.3f} hedged {float((st == 'hedged').mean()):.3f} negated {float((st == 'negated').mean()):.3f} absent {float((st == 'absent').mean()):.3f} | gold prevalence {float(gold[key].mean()):.3f} | "
          f"absent weight {0.1 if key in cfg.under_reported else 0.5} | calibrated absent target per fold " + " ".join(f"{calibs[k]['values'][key]['absent']:.3f}" for k in range(5))
          + f" | gold-train absent cell n/pos fold0 {calibs[0]['counts'][key]['absent']}")

# ---------------------------------------------------------------- 8 metrics
sec("8 metrics thresholds")
y = np.array([0.2, 0.6, 0.6 - 1e-7, 0.2 + 1e-7, 0.5], dtype=np.float32)
w = np.array([0.5, 1.0, 1.0, 0.5, 1.0], dtype=np.float32)
keep, pos = binarise(y, w)
print("float32 [0.2, 0.6, 0.6-1e-7, 0.2+1e-7, 0.5] w [0.5,1,1,0.5,1] -> keep", keep.tolist(), "pos", pos.tolist())
print("  0.2f32 as f64:", float(np.float32(0.2)), " 0.6f32 as f64:", float(np.float32(0.6)))
w2 = np.array([0.5 - 1e-7, 0.49], dtype=np.float32)
keep2, _ = binarise(np.array([0.1, 0.1], dtype=np.float32), w2)
print("  weight float32 0.5 (stored) and 0.49 with target 0.1 -> keep", keep2.tolist())
rng = np.random.default_rng(99)
ys = rng.random((500, 12)); ws = rng.random((500, 12)); ps = ys + rng.normal(0, 0.4, ys.shape)
mac, per = macro_auc(ys, ps, ws)
ref = []
for j in range(12):
    kp = (ys[:, j] >= 0.6) | ((ys[:, j] <= 0.2) & (ws[:, j] >= 0.5))
    ref.append(roc_auc_score((ys[kp, j] >= 0.6).astype(int), ps[kp, j]))
print(f"  macro_auc random case: {mac:.10f} sklearn {np.mean(ref):.10f} diff {abs(mac - np.mean(ref)):.2e}")
# real targets: how many cells sit exactly at a threshold?
tv = t0[[f"{k}_target" for k in LABEL_KEYS]].to_numpy(np.float32)
print("  fold0 target cells exactly 0.2f32:", int((tv == np.float32(0.2)).sum()), "exactly 0.6f32:", int((tv == np.float32(0.6)).sum()),
      "within 1e-6 of 0.2:", int((np.abs(tv - 0.2) < 1e-6).sum()), "of 0.6:", int((np.abs(tv - 0.6) < 1e-6).sum()))
wv = t0[[f"{k}_weight" for k in LABEL_KEYS]].to_numpy(np.float32)
keep_all = np.zeros(tv.shape, bool); pos_all = np.zeros(tv.shape, bool)
for j in range(12):
    keep_all[:, j], pos_all[:, j] = binarise(tv[:, j], wv[:, j])
print("  fold0 OOF-weak binarisation: kept", int(keep_all.sum()), "of", tv.size, "pos", int(pos_all.sum()), "per label pos/neg:",
      {k: (int(pos_all[:, j].sum()), int((keep_all[:, j] & ~pos_all[:, j]).sum())) for j, k in enumerate(LABEL_KEYS)})

# ---------------------------------------------------------------- 9 PAVA hand example
sec("9 PAVA hand examples")
from rsna_kad.labels.targets import pava_decreasing  # noqa: E402
cases = [([0.9, 0.5, 0.6, 0.1], [1, 1, 1, 1], [0.9, 0.55, 0.55, 0.1]),
         ([0.1, 0.9], [1, 3], [0.7, 0.7]),
         ([0.5, 0.6, 0.7, 0.8], [1, 1, 1, 1], [0.65] * 4),
         ([0.9, 0.2, 0.5, 0.4], [1, 1, 3, 1], [0.9, 0.425, 0.425, 0.425]),   # (0.2*1+0.5*3)/4=0.425 then 0.4<0.425 -> pooled (0.2+1.5+0.4)/5 = 0.42
         ([0.9, 0.8, 0.85, 0.1], [2, 1, 1, 2], [0.9, 0.825, 0.825, 0.1])]
for v, w, _ in cases:
    print("  pava", v, w, "->", [round(x, 4) for x in pava_decreasing(v, w)])
print("  expected case4: pooled (0.2*1+0.5*3+0.4*1)/5 =", (0.2 + 1.5 + 0.4) / 5)
# monotone check with sklearn
from sklearn.isotonic import IsotonicRegression  # noqa: E402
rng = np.random.default_rng(3)
n_bad = 0
for _ in range(200):
    v = rng.random(4).tolist(); w = (rng.random(4) * 10 + 1).tolist()
    ref = IsotonicRegression(increasing=False).fit_transform(np.arange(4), v, sample_weight=w)
    if not np.allclose(pava_decreasing(v, w), ref, atol=1e-9):
        n_bad += 1
print("  pava vs sklearn IsotonicRegression(decreasing) on 200 random 4-chains: mismatches", n_bad)
print("\nDONE")
