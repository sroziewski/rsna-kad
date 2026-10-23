#!/usr/bin/env python3
"""Compare label tables produced by different (precision, prompt) configurations on the fixed
A/B subset (qa/llm_ab/eval_ids.txt). Shared venv, CPU only, no LLM.

  qa/llm_ab/compare.py --table awq_base=data/labels/llm_v1_raw.parquet \
                       --table fp8_base=data/labels/llm_bf16_fp8_base.parquet \
                       --baseline awq_base --out qa/llm_ab/comparison

Reports per table: gold-58 macro AUC + bootstrap CI, per-label and per-language AUC, parse rate,
evidence-verbatim rate, throughput, cell-level agreement with the baseline, and counts of the four
error patterns the stage-5 audit named. n=58 makes the gold AUC noisy (+-0.03), so the error
counts and the agreement matrix - computed on all subset rows - carry the decision.

The error detectors are automatable proxies, each anchored on the rule voter's multilingual term
lists (labels/rules.py), so the same definition applies to every language and every table:

  negated_unmentioned  state=negated while the rule voter finds no term for that label anywhere in
                       the report -> the report cannot have negated it ("absent" was right).
                       This is the dominant stage-5 error (14 of 24).
  absent_vs_negated    state=absent while the rule voter reads an explicit negation -> the other
                       direction of the same confusion (missed group statements).
  subthreshold_negated state=negated while the rule voter reads a sub-threshold finding
                       (degeneration / mucoid / sprain / signal change) -> the "degeneration ->
                       negated" violation of the prompt's own rule.
  misattributed        the quote is verbatim but contains no term of this label, while it does
                       contain a term of another label -> the finding was hung on the wrong
                       structure (marrow oedema -> OA, trochlea -> lat_oa).
  quote_missing        non-absent cell whose quote is not in the report (fuzzy ratio < 0.8);
  quote_fabricated     ... < 0.5, i.e. invented rather than paraphrased;
  quote_empty          ... non-absent cell with no quote at all.
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

from rsna_kad.constants import DATA_DIR, LABEL_KEYS, LABELS  # noqa: E402
from rsna_kad.labels import rules as R  # noqa: E402
from rsna_kad.labels.merge import evidence_ratio  # noqa: E402
from rsna_kad.labels.targets import gold_frame, gold_state_report, state_score  # noqa: E402

STATES = ("affirmed", "hedged", "negated", "absent")


# ------------------------------------------------------------------ report-side features


def _key_patterns(lang: str, key: str):
    """Compiled patterns that name this label's structure or finding, incl. its group terms."""
    voc = R._get(lang)
    pats = list(voc.terms.get(key, []))
    for _group, (keys, ps) in voc.both.items():
        if key in keys:
            pats += ps
    if key in R.OA_KEYS:
        pats += voc.oa_cues
    return pats


def report_features(reports: dict[str, str], langs: dict[str, str]) -> dict[str, dict]:
    """Per study: rule mentions/states/severities and, per label, whether a sub-threshold cue sits
    next to one of that label's terms. Computed once and shared by every table."""
    feats: dict[str, dict] = {}
    for sid, text in reports.items():
        lang = langs.get(sid, "en")
        voc = R._get(lang)
        votes, _meta = R.extract_rules_detail(text, lang)
        norm = R.normalize(text)
        sub: dict[str, bool] = {}
        for key in LABEL_KEYS:
            hit = False
            for p in _key_patterns(lang, key):
                for m in p.finditer(norm):
                    lo, hi = max(0, m.start() - 120), min(len(norm), m.end() + 120)
                    if any(q.search(norm[lo:hi]) for q in voc.mild):
                        hit = True
                        break
                if hit:
                    break
            sub[key] = hit
        feats[sid] = {
            "lang": lang, "norm": norm,
            "mentioned": {k: v.mentioned for k, v in votes.items()},
            "rule_state": {k: v.state for k, v in votes.items()},
            "subthreshold": sub,
        }
    return feats


def quote_names_label(quote: str, lang: str, key: str) -> tuple[bool, bool]:
    """(quote names this label, quote names some other label). Empty quote -> (False, False)."""
    if not quote.strip():
        return False, False
    q = R.normalize(quote)
    own = any(p.search(q) for p in _key_patterns(lang, key))
    other = any(any(p.search(q) for p in _key_patterns(lang, k)) for k in LABEL_KEYS if k != key)
    return own, other


# ------------------------------------------------------------------ per-table metrics


def table_metrics(name: str, table: pd.DataFrame, reports: dict[str, str], feats: dict[str, dict],
                  gold: pd.DataFrame, audit_ids: set[str], n_boot: int) -> dict:
    ids = table["StudyInstanceUID"].tolist()
    rep = gold_state_report(table, gold, n_boot=n_boot)
    out: dict = {
        "name": name, "n_rows": len(table), "parse_rate": float(table["parse_ok"].mean()),
        "gold_macro_auc": rep["macro_auc"], "gold_ci95": rep["ci95"], "gold_n": rep["n"],
        "per_label_auc": rep["per_label"],
        "per_lang_auc": {k: {"n": v["n"], "auc": v["macro_auc"]} for k, v in rep.get("per_lang", {}).items()},
    }

    counts = {c: 0 for c in ("cells", "non_absent", "quote_missing", "quote_fabricated", "quote_empty",
                             "negated_unmentioned", "absent_vs_negated", "subthreshold_negated",
                             "misattributed")}
    audit = {k: 0 for k in counts}
    per_label = {k: {c: 0 for c in counts} for k in LABEL_KEYS}
    states = {k: {s: 0 for s in STATES} for k in LABEL_KEYS}
    ratios: list[float] = []
    flagged: list[dict] = []

    for i, sid in enumerate(ids):
        f = feats[sid]
        lang, text = f["lang"], reports.get(sid, "")
        in_audit = sid in audit_ids
        row = table.iloc[i]
        for key in LABEL_KEYS:
            st = str(row[f"{key}_state"])
            ev = str(row[f"{key}_evidence"] or "")
            states[key][st] += 1
            hits = ["cells"]
            if st != "absent":
                hits.append("non_absent")
                ratio = evidence_ratio(ev, text)
                ratios.append(ratio)
                if not ev.strip():
                    hits.append("quote_empty")
                if ratio < 0.8:
                    hits.append("quote_missing")
                if ratio < 0.5:
                    hits.append("quote_fabricated")
                if ratio >= 0.8:
                    own, other = quote_names_label(ev, lang, key)
                    if not own and other:
                        hits.append("misattributed")
            if st == "negated" and not f["mentioned"][key]:
                hits.append("negated_unmentioned")
            if st == "absent" and f["rule_state"][key] == "negated":
                hits.append("absent_vs_negated")
            if st == "negated" and f["subthreshold"][key] and f["rule_state"][key] == "affirmed":
                hits.append("subthreshold_negated")
            for h in hits:
                counts[h] += 1
                per_label[key][h] += 1
                if in_audit:
                    audit[h] += 1
            bad = [h for h in hits if h not in ("cells", "non_absent")]
            if bad and in_audit:
                flagged.append({"study": sid[-8:], "lang": lang, "key": key, "state": st,
                                "flags": bad, "quote": ev[:110]})

    out["counts"] = counts
    out["audit_counts"] = audit
    out["per_label_counts"] = per_label
    out["state_rates"] = {k: {s: v / max(len(ids), 1) for s, v in d.items()} for k, d in states.items()}
    out["evidence_verbatim_rate"] = float(np.mean([r >= 0.8 for r in ratios])) if ratios else float("nan")
    out["evidence_mean_ratio"] = float(np.mean(ratios)) if ratios else float("nan")
    out["audit_flagged_cells"] = flagged
    return out


def gold_head_to_head(a: pd.DataFrame, b: pd.DataFrame, gold: pd.DataFrame) -> dict:
    """Paired test on the gold cells where the two tables disagree: which state is closer to the
    binary gold label under the §2.1 state->score mapping?

    Far more sensitive than the macro AUC on 58 rows, because the 90-95 % of cells the two tables
    agree on contribute nothing but noise to the AUC and are excluded here. `wins` counts cells,
    so a 37-20 split is a sign test with p ~ 0.03 while the AUCs differ by 0.002.
    """
    g = gold.set_index("StudyInstanceUID")
    ai, bi = a.set_index("StudyInstanceUID"), b.set_index("StudyInstanceUID")
    ids = [i for i in g.index if i in ai.index and i in bi.index]
    wins_a = wins_b = 0
    err_a = err_b = 0.0
    n = 0
    per_label: dict[str, list[int]] = {}
    for sid in ids:
        for key in LABEL_KEYS:
            sa, sb = str(ai.loc[sid, f"{key}_state"]), str(bi.loc[sid, f"{key}_state"])
            if sa == sb:
                continue
            y = float(g.loc[sid, key])
            da = abs(state_score(sa, str(ai.loc[sid, f"{key}_severity"]), key) - y)
            db = abs(state_score(sb, str(bi.loc[sid, f"{key}_severity"]), key) - y)
            n += 1
            err_a += da
            err_b += db
            per_label.setdefault(key, [0, 0])
            if da < db - 1e-9:
                wins_a += 1
                per_label[key][0] += 1
            elif db < da - 1e-9:
                wins_b += 1
                per_label[key][1] += 1
    return {"n_disagreeing_gold_cells": n, "n_gold_cells": len(ids) * len(LABEL_KEYS),
            "mean_abs_err_baseline": err_a / max(n, 1), "mean_abs_err_table": err_b / max(n, 1),
            "wins_baseline": wins_a, "wins_table": wins_b,
            "per_label_wins": {k: {"baseline": v[0], "table": v[1]} for k, v in sorted(per_label.items())}}


def agreement(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """Cell-level state agreement between two tables on their common ids."""
    ids = [i for i in a["StudyInstanceUID"] if i in set(b["StudyInstanceUID"])]
    ai, bi = a.set_index("StudyInstanceUID").loc[ids], b.set_index("StudyInstanceUID").loc[ids]
    per_label, same, total = {}, 0, 0
    conf = {f"{x}->{y}": 0 for x in STATES for y in STATES}
    for key in LABEL_KEYS:
        sa = ai[f"{key}_state"].astype(str).to_numpy()
        sb = bi[f"{key}_state"].astype(str).to_numpy()
        eq = sa == sb
        per_label[key] = float(eq.mean())
        same += int(eq.sum())
        total += len(eq)
        for x, y in zip(sa[~eq], sb[~eq]):
            conf[f"{x}->{y}"] += 1
    return {"n_ids": len(ids), "cell_agreement": same / max(total, 1), "per_label": per_label,
            "changes": {k: v for k, v in sorted(conf.items(), key=lambda kv: -kv[1]) if v}}


# ------------------------------------------------------------------ main


def md_table(rows: list[list[str]], header: list[str]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--baseline", default=None, help="name of the reference table for agreement")
    ap.add_argument("--ids", default=str(ROOT / "qa/llm_ab/eval_ids.txt"))
    ap.add_argument("--train", default=f"{DATA_DIR}/train.csv")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", default=str(ROOT / "qa/llm_ab/comparison"))
    args = ap.parse_args(argv)

    ids = [l.strip() for l in Path(args.ids).read_text(encoding="utf-8").splitlines() if l.strip()]
    meta = json.loads(Path(args.ids).with_suffix(".json").read_text(encoding="utf-8"))
    audit_ids = set(meta["audit"])
    train = pd.read_csv(args.train)
    reports = dict(zip(train.StudyInstanceUID, train.Report.fillna("").astype(str)))
    reports = {i: reports[i] for i in ids}
    gold = gold_frame(args.train)

    tables: dict[str, pd.DataFrame] = {}
    for spec in args.table:
        name, _, path = spec.partition("=")
        df = pd.read_parquet(path)
        df = df[df["StudyInstanceUID"].isin(set(ids))].reset_index(drop=True)
        missing = set(ids) - set(df["StudyInstanceUID"])
        if missing:
            print(f"WARNING {name}: {len(missing)} of {len(ids)} subset ids are absent from {path}")
        tables[name] = df
        print(f"{name}: {len(df)} rows from {path}")

    langs = {}
    for df in tables.values():
        langs.update(dict(zip(df["StudyInstanceUID"], df["lang"].astype(str))))
    print("computing rule features ...")
    feats = report_features(reports, langs)

    results = {}
    for name, df in tables.items():
        print(f"metrics for {name} ...")
        results[name] = table_metrics(name, df, reports, feats, gold, audit_ids, args.n_boot)
        stats_path = Path(dict(s.partition("=")[::2] for s in args.table)[name]).with_suffix(".stats.json")
        # A resumed/retry run overwrites .stats.json with the timings of the few re-run reports;
        # the stage-5 full passes kept their real throughput in .stats.fullpass.json.
        full = stats_path.with_suffix(".fullpass.json")
        if full.exists():
            stats_path = full
        if stats_path.exists():
            s = json.loads(stats_path.read_text(encoding="utf-8"))
            results[name]["reports_per_min"] = s.get("reports_per_min")
            results[name]["wall_s"] = s.get("wall_s")
            results[name]["truncated"] = s.get("truncated")
            results[name]["model"] = (s.get("config") or {}).get("model")
            results[name]["prompt_rules"] = (s.get("config") or {}).get("prompt_rules", "base")

    base = args.baseline if args.baseline in tables else next(iter(tables))
    for name, df in tables.items():
        if name != base:
            results[name]["agreement_vs_baseline"] = agreement(tables[base], df)
            results[name]["gold_head_to_head"] = gold_head_to_head(tables[base], df, gold)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(results, indent=1, default=float), encoding="utf-8")

    names = list(tables)
    n_cells = results[names[0]]["counts"]["cells"]
    rows = []
    for n in names:
        r = results[n]
        c, a = r["counts"], r["audit_counts"]
        rows.append([
            n, r.get("model", "?"), r.get("prompt_rules", "?"),
            f"{r['gold_macro_auc']:.3f} [{r['gold_ci95'][0]:.3f},{r['gold_ci95'][1]:.3f}]",
            f"{r['parse_rate']*100:.1f}", f"{r['evidence_verbatim_rate']*100:.1f}",
            f"{r.get('reports_per_min') or float('nan'):.1f}",
            f"{r.get('agreement_vs_baseline', {}).get('cell_agreement', 1.0)*100:.1f}",
            str(c["negated_unmentioned"]), str(c["absent_vs_negated"]), str(c["subthreshold_negated"]),
            str(c["misattributed"]), str(c["quote_missing"]), str(c["quote_fabricated"]),
            f"{a['negated_unmentioned']}/{a['subthreshold_negated']}/{a['misattributed']}/{a['quote_missing']}",
        ])
    header = ["table", "model", "rules", "gold-58 AUC [CI]", "parse%", "verbatim%", "rep/min",
              "cell agr%", "neg_unment", "abs_vs_neg", "subthr_neg", "misattr", "quote_miss",
              "quote_fabr", "audit-25 nu/sn/ma/qm"]
    md = [f"Subset: {len(ids)} studies ({meta['n_gold']} gold, seed-{meta['sample_seed']} sample "
          f"{meta['n_sample']}, audit {meta['n_audit']}); {n_cells} cells per table.", "",
          md_table(rows, header), "",
          "Per-label gold AUC:", "",
          md_table([[n] + [f"{results[n]['per_label_auc'][k]:.3f}" for k in LABEL_KEYS] for n in names],
                   ["table"] + list(LABEL_KEYS)), "",
          "Per-language gold AUC:", ""]
    lang_keys = sorted({l for n in names for l in results[n]["per_lang_auc"]})
    md.append(md_table([[n] + [f"{results[n]['per_lang_auc'][l]['auc']:.3f}" if l in results[n]["per_lang_auc"] else "-"
                               for l in lang_keys] for n in names],
                       ["table"] + [f"{l} (n={results[names[0]]['per_lang_auc'].get(l, {}).get('n', '?')})" for l in lang_keys]))
    md += ["", "Paired gold-cell test (only the gold cells where the two tables disagree; "
               "`wins` = cell is closer to the binary gold label under the §2.1 state->score mapping):", ""]
    h2h_rows = []
    for n in names:
        h = results[n].get("gold_head_to_head")
        if h:
            worst = sorted(h["per_label_wins"].items(), key=lambda kv: kv[1]["baseline"] - kv[1]["table"])
            h2h_rows.append([n, str(h["n_disagreeing_gold_cells"]), f"{h['mean_abs_err_baseline']:.3f}",
                             f"{h['mean_abs_err_table']:.3f}", str(h["wins_baseline"]), str(h["wins_table"]),
                             ", ".join(f"{k} {v['baseline']}-{v['table']}" for k, v in (worst[:1] + worst[-2:]))])
    md.append(md_table(h2h_rows, ["table", "cells", f"mean|err| {base}", "mean|err| table",
                                  f"wins {base}", "wins table", "best / worst labels (base-table)"]))
    for n in names:
        ag = results[n].get("agreement_vs_baseline")
        if ag:
            top = list(ag["changes"].items())[:8]
            md += ["", f"{n} vs {base}: {ag['cell_agreement']*100:.1f} % of cells identical; "
                       f"largest state changes ({base}->{n}): " + ", ".join(f"{k} {v}" for k, v in top)]
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print()
    print("\n".join(md))
    print(f"\nwrote {out.with_suffix('.md')} and {out.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
