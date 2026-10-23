import json

import pytest

from rsna_kad.constants import LABEL_KEYS
from rsna_kad.labels.prompts import (THRESHOLDS, assert_no_gold, build_messages, load_fewshot, system_prompt,
                                     user_prompt)
from rsna_kad.labels.schema import parse_output

HOST_QUOTES = [
    ">50% of fibres disrupted",
    "definitely contacting the meniscal surface on >=2 images",
    "cartilage loss >50% thickness over >=~1 cm",
    "moderate or large only",
    "high-grade partial or complete acute tear",
]


def test_system_prompt_v1_has_host_thresholds_verbatim():
    sp = system_prompt("v1")
    for q in HOST_QUOTES:
        assert q in sp
    for key in LABEL_KEYS:
        assert f"- {key}:" in sp
    assert "affirmed" in sp and "hedged" in sp and "negated" in sp and "absent" in sp


def test_v2_is_report_first_with_checklist():
    up = user_prompt("REPORT TEXT", "v2")
    assert up.index("REPORT TEXT") < up.index("Checklist")
    for q in HOST_QUOTES:
        assert q in up
    assert "Checklist" not in user_prompt("REPORT TEXT", "v1")
    assert system_prompt("v1") != system_prompt("v2")
    with pytest.raises(ValueError):
        system_prompt("v3")


def test_build_messages_structure(tmp_path):
    ex = [{"StudyInstanceUID": f"u{i}", "report": f"report {i}", "output": {"knee_side": "unknown"}} for i in range(3)]
    (tmp_path / "es.json").write_text(json.dumps(ex), encoding="utf-8")
    fs = load_fewshot("es", tmp_path)
    msgs = build_messages("el informe", "es", "v1", fs)
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user", "assistant", "user", "assistant", "user"]
    assert "el informe" in msgs[-1]["content"]
    assert json.loads(msgs[2]["content"]) == {"knee_side": "unknown"}
    # few-shot prefix is identical across reports (prefix cache)
    msgs2 = build_messages("otro informe", "es", "v1", fs)
    assert msgs[:-1] == msgs2[:-1]


def test_load_fewshot_fallback(tmp_path):
    (tmp_path / "en.json").write_text(json.dumps([{"StudyInstanceUID": "e", "report": "r", "output": {}}]), encoding="utf-8")
    assert load_fewshot("fr", tmp_path)[0]["StudyInstanceUID"] == "e"
    assert load_fewshot("fr", tmp_path / "missing") == []
    (tmp_path / "de.json").write_text(json.dumps([{"report": "r"}]), encoding="utf-8")
    with pytest.raises(ValueError):
        load_fewshot("de", tmp_path)


def test_assert_no_gold():
    assert_no_gold(["a", "b"], {"c"})
    with pytest.raises(AssertionError):
        assert_no_gold(["a", "b"], {"b"})


def test_chat_batch_falls_back_to_fewer_shots():
    from types import SimpleNamespace

    from rsna_kad.labels.llm import LLMConfig, _chat_batch

    class FakeLLM:
        def __init__(self):
            self.calls = []

        def chat(self, messages, params, **kw):
            self.calls.append(len(messages))
            outs = []
            for m in messages:
                if sum(len(x["content"]) for x in m if x["role"] != "system") > 450:
                    raise ValueError("maximum context length")
                outs.append(SimpleNamespace(outputs=[SimpleNamespace(text="{}", token_ids=[1], finish_reason="stop")],
                                            prompt_token_ids=[1, 2]))
            return outs

    fewshot = {"en": [{"StudyInstanceUID": f"f{i}", "report": "r" * 60, "output": {}} for i in range(3)]}
    batch = [SimpleNamespace(StudyInstanceUID="a", Report="short", lang="en"),
             SimpleNamespace(StudyInstanceUID="b", Report="x" * 300, lang="en")]
    llm = FakeLLM()
    cfg = LLMConfig()
    outs = _chat_batch(llm, None, cfg, batch, "v1", fewshot)
    assert len(outs) == 2 and outs[0] is not None
    assert llm.calls[0] == 2  # whole batch first, then one by one
    assert outs[1] is not None and outs[1].outputs[0].text == "{}"  # fitted with fewer few-shots
    assert llm.calls == [2, 1, 1, 1, 1]  # batch, a, b(3), b(2), b(1)


def test_repo_fewshots_are_valid_and_not_gold():
    """Every shipped few-shot file: 3 examples, parseable outputs, ids never in the gold set."""
    from rsna_kad.labels.prompts import FEWSHOT_DIR

    files = sorted(FEWSHOT_DIR.glob("??.json")) if FEWSHOT_DIR.exists() else []
    gold_path = FEWSHOT_DIR.parent / "labels" / "gold_ids.txt"
    gold_ids = set(gold_path.read_text().split()) if gold_path.exists() else set()
    for f in files:
        ex = json.loads(f.read_text(encoding="utf-8"))
        assert len(ex) == 3, f
        for e in ex:
            assert parse_output(json.dumps(e["output"])) is not None, f
        assert_no_gold([e["StudyInstanceUID"] for e in ex], gold_ids)
