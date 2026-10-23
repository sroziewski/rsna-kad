"""Collect Claude-subagent outputs into a label table with the same contract as the LLM voters.
    python qa/claude_labels/collect.py --out data/labels/claude_subset.parquet
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rsna_kad.labels.llm import table_from_records
from rsna_kad.labels.schema import write_label_table

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(HERE))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    d = Path(args.dir)

    expected = {}
    for b in json.loads((d / "manifest.json").read_text()):
        for line in (d / "batches" / f"{b['batch']}.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            expected[r["StudyInstanceUID"]] = r["lang"]

    records, bad = {}, []
    for path in sorted((d / "out").glob("*.jsonl")):
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                raw = r["raw"] if isinstance(r["raw"], str) else json.dumps(r["raw"], ensure_ascii=False)
                records[r["StudyInstanceUID"]] = {"StudyInstanceUID": r["StudyInstanceUID"],
                                                  "lang": expected.get(r["StudyInstanceUID"], r.get("lang", "en")),
                                                  "raw": raw}
            except (ValueError, KeyError) as exc:
                bad.append(f"{path.name}:{n} {exc!r}")
    missing = sorted(set(expected) - set(records))
    unexpected = sorted(set(records) - set(expected))
    print(f"expected {len(expected)} | collected {len(records)} | missing {len(missing)} | "
          f"unexpected {len(unexpected)} | unparseable lines {len(bad)}")
    for m in bad[:10]:
        print("  bad:", m)

    table = table_from_records(list(records.values()), extractor="claude-code", variant="v1")
    print(f"parse_ok {int(table['parse_ok'].sum())}/{len(table)}")
    write_label_table(table, args.out) if len(table) == 4407 else table.to_parquet(args.out, index=False)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
