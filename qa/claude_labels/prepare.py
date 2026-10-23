"""Prepare report batches for a Claude label pass run through Claude Code subagents.

Batches are grouped by language so each batch carries one set of same-language few-shots,
and the system prompt is the exact V1/base_qa prompt the Qwen and Gemma voters used.
    python qa/claude_labels/prepare.py --ids qa/llm_ab/eval_ids.txt --batch-size 30
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rsna_kad.labels.llm import load_reports
from rsna_kad.labels.prompts import load_fewshot, system_prompt, user_prompt

HERE = Path(__file__).resolve().parent
VARIANT, RULES = "v1", "base_qa"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", default="/raid/sroziewski/data/rsna_kad/train.csv")
    ap.add_argument("--meta", default="analysis/out/train_meta.csv")
    ap.add_argument("--ids", default=None, help="one StudyInstanceUID per line; default: all studies")
    ap.add_argument("--batch-size", type=int, default=30)
    ap.add_argument("--out", default=str(HERE))
    args = ap.parse_args()

    df = load_reports(args.reports, args.meta)
    if args.ids:
        keep = {l.strip() for l in Path(args.ids).read_text().splitlines() if l.strip()}
        df = df[df["StudyInstanceUID"].isin(keep)]
    out = Path(args.out)
    (out / "batches").mkdir(parents=True, exist_ok=True)
    (out / "out").mkdir(parents=True, exist_ok=True)

    # One prompt file per language: system prompt, then the few-shot pairs, then the
    # template the per-report user turn follows. Subagents read it once per batch.
    for lang in sorted(df["lang"].unique()):
        parts = ["=== SYSTEM ===", system_prompt(VARIANT, RULES), "", "=== EXAMPLES ==="]
        for i, ex in enumerate(load_fewshot(lang), 1):
            parts += [f"--- example {i}: input ---", user_prompt(ex["report"], VARIANT, RULES),
                      f"--- example {i}: output ---", json.dumps(ex["output"], ensure_ascii=False), ""]
        (out / f"prompt_{lang}.txt").write_text("\n".join(parts), encoding="utf-8")

    manifest = []
    for lang, grp in df.sort_values("StudyInstanceUID").groupby("lang"):
        rows = grp.to_dict("records")
        for start in range(0, len(rows), args.batch_size):
            name = f"{lang}_{start // args.batch_size:03d}"
            with open(out / "batches" / f"{name}.jsonl", "w", encoding="utf-8") as fh:
                for r in rows[start:start + args.batch_size]:
                    fh.write(json.dumps({"StudyInstanceUID": r["StudyInstanceUID"], "lang": lang,
                                         "report": r["Report"]}, ensure_ascii=False) + "\n")
            manifest.append({"batch": name, "lang": lang, "n": len(rows[start:start + args.batch_size])})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"{len(df)} reports -> {len(manifest)} batches; languages: {sorted(df['lang'].unique())}")


if __name__ == "__main__":
    main()
