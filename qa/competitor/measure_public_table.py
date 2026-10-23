#!/usr/bin/env python3
"""PATH 2: independently measure the public `pilkwang/rsna-knee-llm-labels` table
(`report_labels_v2.csv`) as a fourth label voter, and test it for the gold-row leak that
research.md §2 says at least five public tables have.

  /raid/sroziewski/sandbox/bin/python qa/competitor/measure_public_table.py \
      --public qa/competitor/llm_labels/report_labels_v2.csv \
      --ours data/labels/llm_v1.parquet --out qa/competitor/public_table

CPU only, no GPU. Writes <out>.json and <out>.md.

What it answers, in order:
  1. coverage + schema      does it cover all 4,407 studies; what do score/conf/verdict contain
                            and how do they map onto our state/severity schema (§2.1).
  2. gold-58 AUC            theirs alone, ours alone, and the rank-average of the two, with a
                            paired bootstrap over studies (§4.2 decision rule).
  3. LEAKAGE                five independent tests of whether the 58 gold rows were written into
                            the table instead of extracted from the reports. If they were, the
                            0.870 is inflated and every comparison must exclude those rows.

The leak tests, from weakest to strongest:
  L1 per-label gold AUC     a report-derived extractor cannot beat the ~18 % report-vs-image
                            disagreement the host documents; AUC ~ 1.00 on any label is a flag.
  L2 verdict accuracy       YES/NO agreement with the binary gold label on gold rows.
  L3 prevalence split       positive rate per label on gold rows vs non-gold rows. Synovitis is
                            gold-positive in 47 % of studies but mentioned in ~12 % of reports
                            (research.md §1), so a table whose gold-row positive rate tracks gold
                            prevalence while its non-gold rate tracks the mention rate is copying.
  L4 agreement split        cell agreement with our llm_v1 states on gold vs non-gold rows: truth
                            differs from the report, so an overwritten gold block agrees *less*.
  L5 unmentioned-affirmed   cells called positive while our multilingual rule voter finds no term
                            for that label anywhere in the report - impossible from the text alone.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rsna_kad.constants import DATA_DIR, ID_COL, LABEL_KEYS, LABELS, LABEL_TO_KEY  # noqa: E402
from rsna_kad.metrics import (  # noqa: E402
    bootstrap_subset_auc, gold_subset, paired_delta, subset_auc,
)
from rsna_kad.labels.targets import gold_frame, label_scores  # noqa: E402

VERDICTS = ("YES", "NO", "UNK")


# ------------------------------------------------------------------ loading


def load_public(path: str | Path) -> pd.DataFrame:
    """Their CSV with our snake-case keys: {key}_score, {key}_conf, {key}_verdict."""
    df = pd.read_csv(path)
    missing = [c for c in LABELS if c not in df.columns]
    if missing:
        raise SystemExit(f"public table lacks label columns: {missing}")
    out = pd.DataFrame({ID_COL: df[ID_COL].astype(str)})
    for label in LABELS:
        key = LABEL_TO_KEY[label]
        out[f"{key}_score"] = df[label].astype(float)
        out[f"{key}_conf"] = df[f"{label}__conf"].astype(float)
        out[f"{key}_verdict"] = df[f"{label}__verdict"].astype(str).str.strip().str.upper()
    return out


def schema_map(pub: pd.DataFrame) -> dict:
    """The distinct (verdict -> score, conf) cells, i.e. how many levels their table really has,
    and the state/severity our schema would assign to each."""
    rows = []
    for key in LABEL_KEYS:
        sub = pd.DataFrame({"verdict": pub[f"{key}_verdict"], "score": pub[f"{key}_score"],
                            "conf": pub[f"{key}_conf"]})
        for (v, s, c), n in sub.value_counts().items():
            rows.append({"key": key, "verdict": v, "score": float(s), "conf": float(c), "n": int(n)})
    cells = pd.DataFrame(rows)
    glob = cells.groupby(["verdict", "score", "conf"], as_index=False)["n"].sum().sort_values("n", ascending=False)
    return {
        "n_distinct_cells_global": int(len(glob)),
        "cells": glob.to_dict("records"),
        "verdict_counts": {v: int((pub[[f"{k}_verdict" for k in LABEL_KEYS]] == v).to_numpy().sum())
                           for v in VERDICTS},
        "score_is_function_of_verdict": bool(glob.groupby("verdict")["score"].nunique().max() <= 2),
    }


# ------------------------------------------------------------------ predictors


def public_pred(pub: pd.DataFrame) -> pd.DataFrame:
    """{key}_pred from their `score` column - the number their own pipeline uses as the target."""
    out = pd.DataFrame({ID_COL: pub[ID_COL].to_numpy()})
    for key in LABEL_KEYS:
        out[f"{key}_pred"] = pub[f"{key}_score"].to_numpy(dtype=np.float64)
    return out


def rank01(x: np.ndarray) -> np.ndarray:
    """Average-rank transform to [0,1]; ties share a rank (their score has ~3 levels per label,
    so tie handling is what decides whether the rank-average is meaningful at all)."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < 2:
        return np.zeros(n)
    order = np.argsort(x, kind="mergesort")
    s = x[order]
    new = np.empty(n, dtype=bool)
    new[0] = True
    np.not_equal(s[1:], s[:-1], out=new[1:])
    group = np.cumsum(new) - 1
    starts = np.flatnonzero(new)
    ends = np.append(starts[1:], n) - 1
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = (starts[group] + ends[group]) / 2.0
    return ranks / (n - 1)


def rank_average(preds: list[pd.DataFrame], ids: list[str]) -> pd.DataFrame:
    """Per-label mean of the rank01 transforms, ranked over `ids` (the shared population).

    Ranking over the whole corpus rather than over the 57 gold rows is the operational choice:
    the voter set has to produce targets for 4,349 studies, and a within-gold ranking would be a
    different (and unusable) estimator."""
    aligned = [p.set_index(ID_COL).loc[ids] for p in preds]
    out = pd.DataFrame({ID_COL: ids})
    for key in LABEL_KEYS:
        out[f"{key}_pred"] = np.mean([rank01(a[f"{key}_pred"].to_numpy()) for a in aligned], axis=0)
    return out


# ------------------------------------------------------------------ leakage tests


def positive_rate(pub: pd.DataFrame, mask: np.ndarray) -> dict[str, float]:
    return {k: float((pub.loc[mask, f"{k}_verdict"] == "YES").mean()) for k in LABEL_KEYS}


def state_agreement(pub: pd.DataFrame, ours: pd.DataFrame, ids: list[str]) -> dict:
    """Cell agreement between their verdict and our llm_v1 state, on `ids`.

    Mapping for the comparison only: YES<->affirmed|hedged, NO<->negated, UNK<->absent."""
    p = pub.set_index(ID_COL).loc[ids]
    o = ours.set_index(ID_COL).loc[ids]
    per_label, same, total = {}, 0, 0
    for key in LABEL_KEYS:
        st = o[f"{key}_state"].astype(str).to_numpy()
        mapped = np.where(np.isin(st, ["affirmed", "hedged"]), "YES",
                          np.where(st == "negated", "NO", "UNK"))
        eq = mapped == p[f"{key}_verdict"].to_numpy()
        per_label[key] = float(eq.mean())
        same += int(eq.sum())
        total += len(eq)
    return {"n_ids": len(ids), "cell_agreement": same / max(total, 1), "per_label": per_label}


def unmentioned_positive(pub: pd.DataFrame, reports: dict[str, str], langs: dict[str, str],
                         ids: list[str]) -> dict:
    """Rate of YES cells for which the rule voter finds no term of that label in the report."""
    from rsna_kad.labels import rules as R

    p = pub.set_index(ID_COL)
    n_yes = n_bad = 0
    per_label = {k: [0, 0] for k in LABEL_KEYS}
    for sid in ids:
        votes, _meta = R.extract_rules_detail(reports.get(sid, ""), langs.get(sid, "en"))
        for key in LABEL_KEYS:
            if str(p.loc[sid, f"{key}_verdict"]) != "YES":
                continue
            n_yes += 1
            per_label[key][0] += 1
            if not votes[key].mentioned:
                n_bad += 1
                per_label[key][1] += 1
    return {"n_ids": len(ids), "n_yes": n_yes, "n_yes_unmentioned": n_bad,
            "rate": n_bad / max(n_yes, 1),
            "per_label": {k: {"yes": v[0], "unmentioned": v[1],
                              "rate": v[1] / max(v[0], 1)} for k, v in per_label.items()}}


# ------------------------------------------------------------------ main


def fmt(x: float, n: int = 4) -> str:
    return "nan" if x is None or not np.isfinite(x) else f"{x:.{n}f}"


def md_table(rows, header) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--public", default=str(ROOT / "qa/competitor/llm_labels/report_labels_v2.csv"))
    ap.add_argument("--ours", default=str(ROOT / "data/labels/llm_v1.parquet"))
    ap.add_argument("--train", default=f"{DATA_DIR}/train.csv")
    ap.add_argument("--meta", default=str(ROOT / "analysis/out/train_meta.csv"))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-nongold-sample", type=int, default=600,
                    help="non-gold studies used for the rule-voter leak test (L5)")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=str(ROOT / "qa/competitor/public_table"))
    args = ap.parse_args(argv)

    train = pd.read_csv(args.train)
    all_ids = set(train[ID_COL].astype(str))
    reports = dict(zip(train[ID_COL].astype(str), train["Report"].fillna("").astype(str)))
    gold = gold_frame(args.train)
    gold_ids = set(gold[ID_COL].astype(str))

    pub = load_public(args.public)
    ours = pd.read_parquet(args.ours)
    ours[ID_COL] = ours[ID_COL].astype(str)

    res: dict = {}

    # ---------------------------------------------------------- 1. coverage + schema
    pub_ids = set(pub[ID_COL])
    res["coverage"] = {
        "public_rows": int(len(pub)), "public_unique_ids": int(len(pub_ids)),
        "train_studies": int(len(all_ids)),
        "covered": int(len(pub_ids & all_ids)),
        "missing_from_public": sorted(all_ids - pub_ids)[:20],
        "n_missing_from_public": int(len(all_ids - pub_ids)),
        "n_extra_in_public": int(len(pub_ids - all_ids)),
        "gold_rows_present": int(len(pub_ids & gold_ids)), "gold_rows_total": int(len(gold_ids)),
        "nan_cells": int(pub[[f"{k}_score" for k in LABEL_KEYS]].isna().to_numpy().sum()),
    }
    res["schema"] = schema_map(pub)

    # ---------------------------------------------------------- 2. gold-58 AUC
    shared = sorted(pub_ids & set(ours[ID_COL]))
    res["n_shared"] = len(shared)
    p_pub = public_pred(pub)
    p_ours = label_scores(ours)
    p_ours[ID_COL] = p_ours[ID_COL].astype(str)
    p_avg = rank_average([p_ours, p_pub], shared)

    gold_shared = gold[gold[ID_COL].astype(str).isin(set(shared))].reset_index(drop=True)
    gs = gold_subset(gold_shared)
    cand = {"ours_llm_v1": p_ours, "public_pilkwang": p_pub, "rank_avg(ours+public)": p_avg}
    aligned = {n: gs.align(p) for n, p in cand.items()}
    res["gold_auc"] = {}
    for name, arr in aligned.items():
        macro, per_label = subset_auc(gs, arr)
        res["gold_auc"][name] = {"macro": macro, "per_label": per_label}
    res["gold_n"] = int(len(gold_shared))

    boots = bootstrap_subset_auc(gs, aligned, n_boot=args.n_boot, seed=args.seed)
    res["paired"] = {
        "rank_avg_vs_ours": paired_delta(boots["rank_avg(ours+public)"], boots["ours_llm_v1"]),
        "public_vs_ours": paired_delta(boots["public_pilkwang"], boots["ours_llm_v1"]),
    }

    # per-label Pearson correlation of the two extractors over the shared corpus
    a = p_ours.set_index(ID_COL).loc[shared]
    b = p_pub.set_index(ID_COL).loc[shared]
    corr = {}
    for key in LABEL_KEYS:
        x, y = a[f"{key}_pred"].to_numpy(float), b[f"{key}_pred"].to_numpy(float)
        corr[key] = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else float("nan")
    res["extractor_correlation"] = {"per_label": corr,
                                    "mean": float(np.nanmean(list(corr.values())))}

    # ---------------------------------------------------------- 3. leakage
    g_ids = sorted(gold_ids & pub_ids)
    ng_ids = sorted(pub_ids & all_ids - gold_ids)
    is_gold = pub[ID_COL].isin(gold_ids).to_numpy()

    leak: dict = {}
    leak["L1_per_label_gold_auc"] = res["gold_auc"]["public_pilkwang"]["per_label"]
    leak["L1_n_labels_auc_ge_0_99"] = int(sum(
        1 for v in leak["L1_per_label_gold_auc"].values() if np.isfinite(v) and v >= 0.99))

    gi = gold.set_index(ID_COL)
    pi = pub.set_index(ID_COL)
    acc, n_cells, conf_rows = 0, 0, []
    for key in LABEL_KEYS:
        v = pi.loc[g_ids, f"{key}_verdict"].to_numpy()
        y = gi.loc[g_ids, key].to_numpy(dtype=float) >= 0.5
        pred_pos, pred_neg = v == "YES", v == "NO"
        hit = int((pred_pos & y).sum() + (pred_neg & ~y).sum())
        dec = int((pred_pos | pred_neg).sum())
        acc += hit
        n_cells += dec
        conf_rows.append({"key": key, "n_decided": dec, "accuracy": hit / max(dec, 1),
                          "unk_rate": float((v == "UNK").mean())})
    leak["L2_verdict_vs_gold"] = {"n_decided_cells": n_cells,
                                  "accuracy_on_decided": acc / max(n_cells, 1),
                                  "per_label": conf_rows}

    pos_gold = positive_rate(pub, is_gold)
    pos_ng = positive_rate(pub, ~is_gold)
    gold_prev = {k: float((gi.loc[g_ids, k].to_numpy(dtype=float) >= 0.5).mean()) for k in LABEL_KEYS}
    # Control: the gold 58 are an *enriched* sample (architecture A§Implications 1), so any honest
    # extractor shows a positive-rate gap on them. Our own llm_v1 is the null distribution for L3 -
    # without it the test cannot tell copying from enrichment.
    og = ours.set_index(ID_COL)
    ours_pos_gold = {k: float(og.loc[g_ids, f"{k}_state"].astype(str).isin(["affirmed", "hedged"]).mean())
                     for k in LABEL_KEYS}
    ours_pos_ng = {k: float(og.loc[ng_ids, f"{k}_state"].astype(str).isin(["affirmed", "hedged"]).mean())
                   for k in LABEL_KEYS}
    ours_mean_gap = float(np.mean([abs(ours_pos_gold[k] - ours_pos_ng[k]) for k in LABEL_KEYS]))
    leak["L3_positive_rate"] = {"gold_rows": pos_gold, "non_gold_rows": pos_ng,
                                "gold_prevalence": gold_prev,
                                "max_abs_gap": max(abs(pos_gold[k] - pos_ng[k]) for k in LABEL_KEYS),
                                "mean_abs_gap": float(np.mean([abs(pos_gold[k] - pos_ng[k]) for k in LABEL_KEYS])),
                                "control_ours_gold_rows": ours_pos_gold,
                                "control_ours_non_gold": ours_pos_ng,
                                "control_ours_mean_abs_gap": ours_mean_gap}

    rng = np.random.default_rng(args.seed)
    ng_sample = sorted(rng.choice(ng_ids, min(args.n_nongold_sample, len(ng_ids)), replace=False).tolist())
    leak["L4_agreement_with_ours"] = {
        "gold_rows": state_agreement(pub, ours, g_ids),
        "non_gold_sample": state_agreement(pub, ours, ng_sample),
    }
    leak["L4_agreement_with_ours"]["gap"] = (
        leak["L4_agreement_with_ours"]["non_gold_sample"]["cell_agreement"]
        - leak["L4_agreement_with_ours"]["gold_rows"]["cell_agreement"])

    langs = dict(zip(ours[ID_COL], ours["lang"].astype(str)))
    leak["L5_unmentioned_positive"] = {
        "gold_rows": unmentioned_positive(pub, reports, langs, g_ids),
        "non_gold_sample": unmentioned_positive(pub, reports, langs, ng_sample),
    }
    leak["L5_unmentioned_positive"]["gap"] = (
        leak["L5_unmentioned_positive"]["gold_rows"]["rate"]
        - leak["L5_unmentioned_positive"]["non_gold_sample"]["rate"])
    res["leakage"] = leak

    # A verdict is a leak only if several tests point the same way; state the call explicitly.
    votes = [
        leak["L1_n_labels_auc_ge_0_99"] >= 1,
        leak["L2_verdict_vs_gold"]["accuracy_on_decided"] >= 0.95,
        leak["L3_positive_rate"]["mean_abs_gap"] >= leak["L3_positive_rate"]["control_ours_mean_abs_gap"] + 0.10,
        leak["L4_agreement_with_ours"]["gap"] >= 0.05,
        leak["L5_unmentioned_positive"]["gap"] >= 0.10,
    ]
    res["leak_verdict"] = {"flags": int(sum(votes)), "tests": votes,
                           "call": "LEAKS GOLD" if sum(votes) >= 3 else
                                   ("SUSPICIOUS" if sum(votes) == 2 else "no evidence of a gold leak")}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")

    md = [f"# PATH 2 - public table `report_labels_v2.csv` measured ({res['coverage']['public_rows']} rows)", "",
          f"Coverage: {res['coverage']['covered']}/{res['coverage']['train_studies']} train studies; "
          f"{res['coverage']['n_missing_from_public']} missing, {res['coverage']['n_extra_in_public']} extra; "
          f"gold rows present {res['coverage']['gold_rows_present']}/{res['coverage']['gold_rows_total']}; "
          f"NaN cells {res['coverage']['nan_cells']}.", "",
          f"Schema: {res['schema']['n_distinct_cells_global']} distinct (verdict, score, conf) cells; "
          f"verdict counts {res['schema']['verdict_counts']}.", "",
          md_table([[c["verdict"], fmt(c["score"], 2), fmt(c["conf"], 2), c["n"]]
                    for c in res["schema"]["cells"][:10]], ["verdict", "score", "conf", "cells"]), "",
          f"## gold-{res['gold_n']} macro AUC (shared rows: {res['n_shared']})", "",
          md_table([[n, fmt(v["macro"])] for n, v in res["gold_auc"].items()], ["voter set", "macro AUC"]), "",
          "Per-label:", "",
          md_table([[n] + [fmt(v["per_label"][k], 3) for k in LABEL_KEYS]
                    for n, v in res["gold_auc"].items()], ["voter set"] + list(LABEL_KEYS)), "",
          "Paired bootstrap over gold studies (§4.2):", "",
          md_table([[n, fmt(d["mean"]), f"[{fmt(d['ci_lo'])}, {fmt(d['ci_hi'])}]", fmt(d["p_gt"], 3)]
                    for n, d in res["paired"].items()], ["delta", "mean", "CI95", "P(>0)"]), "",
          f"Extractor correlation (per-label Pearson over {res['n_shared']} studies): "
          f"mean {fmt(res['extractor_correlation']['mean'], 3)}; "
          + ", ".join(f"{k} {fmt(v, 3)}" for k, v in sorted(corr.items(), key=lambda kv: kv[1])[:3]), "",
          "## Gold-leak tests", "",
          f"- **L1** labels with gold AUC >= 0.99: {leak['L1_n_labels_auc_ge_0_99']} "
          f"({', '.join(f'{k} {fmt(v, 3)}' for k, v in sorted(leak['L1_per_label_gold_auc'].items(), key=lambda kv: -kv[1])[:3])})",
          f"- **L2** verdict accuracy on decided gold cells: {fmt(leak['L2_verdict_vs_gold']['accuracy_on_decided'], 3)} "
          f"over {leak['L2_verdict_vs_gold']['n_decided_cells']} cells",
          f"- **L3** mean |positive-rate gap| gold vs non-gold rows: {fmt(leak['L3_positive_rate']['mean_abs_gap'], 3)} "
          f"(max {fmt(leak['L3_positive_rate']['max_abs_gap'], 3)}); control - our own llm_v1 on the same rows: "
          f"{fmt(leak['L3_positive_rate']['control_ours_mean_abs_gap'], 3)}",
          f"- **L4** cell agreement with our llm_v1: gold {fmt(leak['L4_agreement_with_ours']['gold_rows']['cell_agreement'], 3)} "
          f"vs non-gold {fmt(leak['L4_agreement_with_ours']['non_gold_sample']['cell_agreement'], 3)} "
          f"(gap {fmt(leak['L4_agreement_with_ours']['gap'], 3)})",
          f"- **L5** YES cells with no rule-voter mention: gold {fmt(leak['L5_unmentioned_positive']['gold_rows']['rate'], 3)} "
          f"vs non-gold {fmt(leak['L5_unmentioned_positive']['non_gold_sample']['rate'], 3)} "
          f"(gap {fmt(leak['L5_unmentioned_positive']['gap'], 3)})", "",
          f"**Call: {res['leak_verdict']['call']}** ({res['leak_verdict']['flags']}/5 tests flag a leak).", "",
          md_table([[k, fmt(pos_gold[k], 3), fmt(pos_ng[k], 3), fmt(ours_pos_gold[k], 3),
                     fmt(ours_pos_ng[k], 3), fmt(gold_prev[k], 3)] for k in LABEL_KEYS],
                   ["label", "public YES (gold rows)", "public YES (non-gold)",
                    "ours pos (gold rows)", "ours pos (non-gold)", "gold prevalence"]),
          ]
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"\nwrote {out.with_suffix('.md')} and {out.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
