"""LLM extractor on vLLM (runs in .venv-llm). Results stream to a jsonl next to the output so an
interrupted run resumes where it stopped; the parquet/csv table is rebuilt from the jsonl."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml

from rsna_kad.constants import HF_HOME, LABELS
from rsna_kad.labels.merge import add_evidence_flags
from rsna_kad.labels.prompts import assert_no_gold, build_messages, load_fewshot
from rsna_kad.labels.schema import JSON_SCHEMA, label_row, parse_output, write_label_table

log = logging.getLogger("rsna_kad.llm")


@dataclass
class LLMConfig:
    model: str = "Qwen/Qwen3-32B-AWQ"
    quantization: str | None = "awq_marlin"
    dtype: str = "float16"
    max_model_len: int = 8192
    gpu_memory_utilization: float = 0.6
    enable_prefix_caching: bool = True
    max_tokens: int = 900
    temperature: float = 0.0
    batch_size: int = 128
    seed: int = 0
    enable_thinking: bool = False
    enforce_eager: bool = False
    max_num_seqs: int = 64
    fewshot_dir: str = "data/fewshot"
    out_dir: str = "data/labels"
    extractor: str = "qwen3_32b_awq"
    # Serving modes beyond one-GPU int4; defaults reproduce the stage-5 run exactly.
    tensor_parallel_size: int = 1
    cpu_offload_gb: float = 0.0
    max_num_batched_tokens: int | None = None
    prompt_rules: str = "base"        # prompts.PROMPT_RULES: "base" (stage 5) or "strict"

    @classmethod
    def load(cls, path: str | Path) -> "LLMConfig":
        doc = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        unknown = set(doc) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown keys in {path}: {sorted(unknown)}")
        return cls(**doc)


def load_reports(reports_csv: str | Path, meta_csv: str | Path | None = None) -> pd.DataFrame:
    """StudyInstanceUID, Report, lang, is_gold. Language from the analyst's train_meta.csv when
    available, otherwise langdetect with seed 0 (data_analysis.md §1)."""
    df = pd.read_csv(reports_csv)
    df["is_gold"] = df[LABELS[0]].notna()
    df = df[["StudyInstanceUID", "Report", "is_gold"]].copy()
    df["Report"] = df["Report"].fillna("").astype(str)
    lang = None
    if meta_csv and Path(meta_csv).exists():
        meta = pd.read_csv(meta_csv)
        if "lang" in meta.columns:
            lang = dict(zip(meta["StudyInstanceUID"], meta["lang"].astype(str)))
    if lang is None:
        from langdetect import DetectorFactory, detect

        DetectorFactory.seed = 0
        lang = {}
        for sid, rep in zip(df["StudyInstanceUID"], df["Report"]):
            try:
                lang[sid] = detect(rep)
            except Exception:  # noqa: BLE001 - langdetect raises its own base error on empty text
                lang[sid] = "en"
    df["lang"] = df["StudyInstanceUID"].map(lang).fillna("en").astype(str)
    return df


def _read_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    done: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return done
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue  # a torn last line from an interrupted run
            done[rec["StudyInstanceUID"]] = rec
    return done


def _batches(items: list[Any], n: int) -> Iterable[list[Any]]:
    for i in range(0, len(items), n):
        yield items[i:i + n]


def build_llm(cfg: LLMConfig):
    from vllm import LLM

    kwargs: dict[str, Any] = dict(
        model=cfg.model, dtype=cfg.dtype, max_model_len=cfg.max_model_len,
        gpu_memory_utilization=cfg.gpu_memory_utilization, enable_prefix_caching=cfg.enable_prefix_caching,
        seed=cfg.seed, enforce_eager=cfg.enforce_eager, max_num_seqs=cfg.max_num_seqs, download_dir=HF_HOME,
    )
    if cfg.quantization:
        kwargs["quantization"] = cfg.quantization
    if cfg.tensor_parallel_size != 1:
        kwargs["tensor_parallel_size"] = cfg.tensor_parallel_size
    if cfg.cpu_offload_gb:
        kwargs["cpu_offload_gb"] = cfg.cpu_offload_gb
    if cfg.max_num_batched_tokens:
        kwargs["max_num_batched_tokens"] = cfg.max_num_batched_tokens
    return LLM(**kwargs)


def sampling_params(cfg: LLMConfig):
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    return SamplingParams(temperature=cfg.temperature, max_tokens=cfg.max_tokens, seed=cfg.seed,
                          structured_outputs=StructuredOutputsParams(json=JSON_SCHEMA))


def _chat(llm, params, cfg: LLMConfig, messages: list[list[dict[str, str]]]):
    return llm.chat(messages, params, use_tqdm=False,
                    chat_template_kwargs={"enable_thinking": cfg.enable_thinking})


def _chat_batch(llm, params, cfg: LLMConfig, batch, variant: str, fewshots: dict[str, list]):
    """One batch; a prompt that exceeds the context (long report + long few-shots) is retried
    alone with fewer few-shots, and gives None if it still does not fit."""
    messages = [build_messages(r.Report, r.lang, variant, fewshots[r.lang], cfg.prompt_rules) for r in batch]
    try:
        return _chat(llm, params, cfg, messages)
    except Exception as exc:  # noqa: BLE001 - vLLM raises its own validation error class
        log.warning("batch failed (%s); retrying items one by one", str(exc)[:160])
    outputs = []
    for r in batch:
        out = None
        for n_shots in (len(fewshots[r.lang]), 2, 1, 0):
            try:
                out = _chat(llm, params, cfg, [build_messages(r.Report, r.lang, variant,
                                                              fewshots[r.lang][:n_shots], cfg.prompt_rules)])[0]
                break
            except Exception as exc:  # noqa: BLE001
                log.warning("%s with %d few-shots failed: %s", r.StudyInstanceUID, n_shots, str(exc)[:120])
        outputs.append(out)
    return outputs


def table_from_records(records: Iterable[dict[str, Any]], extractor: str, variant: str,
                       reports: dict[str, str] | None = None) -> pd.DataFrame:
    rows = [label_row(r["StudyInstanceUID"], r["lang"], extractor, variant, parse_output(r["raw"])) for r in records]
    df = pd.DataFrame(rows)
    if reports is not None:
        df = add_evidence_flags(df, reports)
    return df


def run_extractor(reports_csv: str | Path, cfg: LLMConfig, variant: str, out_path: str | Path,
                  meta_csv: str | Path | None = None, limit: int | None = None,
                  ids: list[str] | None = None, use_fewshot: bool = True, llm=None,
                  retry_failed: bool = False) -> dict[str, Any]:
    """Run one prompt variant over all reports (resumable), write the label table and a stats json.
    retry_failed drops unparseable/truncated results from the jsonl first so they run again."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_path.with_suffix(".jsonl")
    reports = load_reports(reports_csv, meta_csv)
    gold_ids = set(reports.loc[reports["is_gold"], "StudyInstanceUID"])
    fewshots: dict[str, list[dict[str, Any]]] = {}
    for lang in sorted(reports["lang"].unique()):
        fewshots[lang] = load_fewshot(lang, cfg.fewshot_dir) if use_fewshot else []
        assert_no_gold([ex["StudyInstanceUID"] for ex in fewshots[lang]], gold_ids)
    if ids is not None:
        reports = reports[reports["StudyInstanceUID"].isin(set(ids))]
    if limit:
        reports = reports.head(limit)

    done = _read_jsonl(jsonl_path)
    if retry_failed:
        failed = [sid for sid, rec in done.items() if not rec.get("parse_ok") or rec.get("finish") != "stop"]
        if failed:
            for sid in failed:
                done.pop(sid)
            with open(jsonl_path, "w", encoding="utf-8") as fh:
                for rec in done.values():
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            log.info("variant %s: %d failed/truncated results dropped for a retry", variant, len(failed))
    todo = reports[~reports["StudyInstanceUID"].isin(done)].sort_values("lang", kind="stable")
    log.info("variant %s: %d reports, %d already done, %d to run", variant, len(reports), len(done), len(todo))

    stats: dict[str, Any] = {"variant": variant, "n_total": int(len(reports)), "n_resumed": int(len(done))}
    if len(todo):
        if llm is None:
            llm = build_llm(cfg)
        params = sampling_params(cfg)
        t0 = time.time()
        n_done = 0
        gen_tokens = prompt_tokens = 0
        with open(jsonl_path, "a", encoding="utf-8") as fh:
            for batch in _batches(list(todo.itertuples(index=False)), cfg.batch_size):
                tb = time.time()
                outputs = _chat_batch(llm, params, cfg, batch, variant, fewshots)
                for r, out in zip(batch, outputs):
                    text = out.outputs[0].text if out is not None else ""
                    rec = {"StudyInstanceUID": r.StudyInstanceUID, "lang": r.lang, "variant": variant,
                           "raw": text, "parse_ok": parse_output(text) is not None,
                           "prompt_tokens": len(out.prompt_token_ids or []) if out is not None else 0,
                           "gen_tokens": len(out.outputs[0].token_ids) if out is not None else 0,
                           "finish": out.outputs[0].finish_reason if out is not None else "error"}
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    gen_tokens += rec["gen_tokens"]
                    prompt_tokens += rec["prompt_tokens"]
                    done[r.StudyInstanceUID] = rec
                fh.flush()
                n_done += len(batch)
                el = time.time() - t0
                log.info("%s: %d/%d done, batch %.0fs, %.1f reports/min, %.0f gen tok/s",
                         variant, n_done, len(todo), time.time() - tb, 60 * n_done / max(el, 1e-6), gen_tokens / max(el, 1e-6))
        wall = time.time() - t0
        stats.update({"n_run": n_done, "wall_s": wall, "reports_per_min": 60 * n_done / max(wall, 1e-6),
                      "gen_tokens": gen_tokens, "prompt_tokens": prompt_tokens})

    records = [done[sid] for sid in reports["StudyInstanceUID"] if sid in done]
    rep_text = dict(zip(reports["StudyInstanceUID"], reports["Report"]))
    table = table_from_records(records, cfg.extractor, variant, rep_text)
    write_label_table(table, out_path)
    stats["parse_rate"] = float(table["parse_ok"].mean()) if len(table) else float("nan")
    stats["n_rows"] = int(len(table))
    stats["truncated"] = int(sum(1 for r in records if r.get("finish") == "length"))
    ok_cells = n_cells = 0
    for key in [c[:-6] for c in table.columns if c.endswith("_state")]:
        m = table[f"{key}_state"].astype(str).isin(["affirmed", "negated", "hedged"])
        n_cells += int(m.sum())
        ok_cells += int(table.loc[m, f"{key}_evidence_ok"].sum())
    stats["evidence_ok_rate"] = ok_cells / n_cells if n_cells else float("nan")
    stats["config"] = asdict(cfg)
    out_path.with_suffix(".stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    log.info("wrote %s (%d rows, parse %.4f, evidence %.4f)", out_path, len(table), stats["parse_rate"], stats["evidence_ok_rate"])
    return stats


def draft_fewshot(reports_csv: str | Path, cfg: LLMConfig, ids: list[str], meta_csv: str | Path | None,
                  variant: str = "v1", llm=None) -> list[dict[str, Any]]:
    """Zero-shot drafts for hand-editing into few-shot files (never gold rows)."""
    reports = load_reports(reports_csv, meta_csv)
    gold_ids = set(reports.loc[reports["is_gold"], "StudyInstanceUID"])
    assert_no_gold(ids, gold_ids)
    sub = reports.set_index("StudyInstanceUID").loc[ids]
    if llm is None:
        llm = build_llm(cfg)
    messages = [build_messages(r.Report, r.lang, variant, [], cfg.prompt_rules) for r in sub.itertuples()]
    outputs = llm.chat(messages, sampling_params(cfg), use_tqdm=False,
                       chat_template_kwargs={"enable_thinking": cfg.enable_thinking})
    drafts = []
    for sid, r, out in zip(ids, sub.itertuples(), outputs):
        parsed = parse_output(out.outputs[0].text)
        drafts.append({"StudyInstanceUID": sid, "lang": r.lang, "report": r.Report,
                       "output": parsed.to_dict() if parsed else None, "raw": out.outputs[0].text})
    return drafts
