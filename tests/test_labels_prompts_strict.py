"""The `strict` rule set added for the qa/llm_ab A/B test (prompts.PROMPT_RULES).

It is an alternative wording of the same schema, selected by `LLMConfig.prompt_rules`; the
default must stay byte-identical to the stage-5 prompts that produced data/labels/llm_v1*.
"""
import pytest

from rsna_kad.labels.llm import LLMConfig
from rsna_kad.labels.prompts import PROMPT_RULES, build_messages, system_prompt, user_prompt


def test_default_rules_leave_the_stage5_prompts_untouched():
    for variant in ("v1", "v2"):
        assert system_prompt(variant) == system_prompt(variant, "base")
        assert user_prompt("REPORT", variant) == user_prompt("REPORT", variant, "base")
    assert build_messages("R", "en", "v1", []) == build_messages("R", "en", "v1", [], "base")
    assert LLMConfig().prompt_rules == "base"


def test_strict_rules_differ_and_cover_the_audited_failure_modes():
    sp = system_prompt("v1", "strict")
    assert sp != system_prompt("v1", "base")
    # group statements name both meniscus labels; "no internal derangement" is scoped explicitly
    assert "med_men and lat_men" in sp.replace("BOTH ", "")
    assert "internal derangement" in sp
    # sub-threshold findings stay affirmed/mild, structure attribution and verbatim quoting are rules
    assert "degeneration" in sp and "never \"negated\"" in sp
    assert "trochlea" in sp
    assert "character for character" in sp
    # the host thresholds and the 12 keys survive the rewrite
    assert ">50% of fibres disrupted" in sp
    assert all(f"- {k}:" in sp for k in ("acl", "mcl", "med_men", "lat_men", "effusion", "fracture"))
    # v2 keeps its report-first layout; the rules live in the checklist
    assert "STEP 1" in user_prompt("REPORT", "v2", "strict")
    assert user_prompt("REPORT", "v2", "strict").index("REPORT") < user_prompt("REPORT", "v2", "strict").index("Checklist")


def test_unknown_rule_set_raises_on_both_variants():
    assert PROMPT_RULES == ("base", "strict", "strict2", "base_qa")
    for variant in ("v1", "v2"):
        with pytest.raises(ValueError):
            system_prompt(variant, "v3")
        with pytest.raises(ValueError):
            user_prompt("R", variant, "v3")


def test_strict2_only_changes_the_step1_step3_precedence():
    """`strict` produced data/labels/llm_bf16_awq_strict.*; its wording must stay reproducible."""
    a, b = system_prompt("v1", "strict"), system_prompt("v1", "strict2")
    assert a != b
    assert "STEP 3 overrides this" in b and "STEP 3 overrides this" not in a
    assert "no joint space narrowing" in b and "no joint space narrowing" not in a
    # everything the strict rewrite was for survives in strict2
    for must in ("character for character", "trochlea", "internal derangement", "degeneration"):
        assert must in b
    # md5 of the `strict` text, pinned so a later edit cannot silently invalidate the A/B table
    import hashlib

    assert hashlib.md5(a.encode()).hexdigest() == "a036bd85a1265e03a7080e92f5a0878f"


def test_strict_prefix_is_still_shared_across_reports():
    """Prefix caching pays for the longer rule text only if everything but the report is constant."""
    fs = [{"StudyInstanceUID": "u", "report": "r", "output": {}}]
    a = build_messages("report one", "en", "v1", fs, "strict")
    b = build_messages("report two", "en", "v1", fs, "strict")
    assert a[:-1] == b[:-1]


def test_base_qa_is_base_plus_the_two_licensed_paragraphs():
    """targets v1.1 carry-forward: the A/B licensed the `strict` quoting and attribution
    paragraphs and nothing else, because its STEP 1/STEP 2 state rules lost on gold."""
    from rsna_kad.labels.prompts import STATE_RULES, STATE_RULES_BASE_QA

    sp = system_prompt("v1", "base_qa")
    assert sp != system_prompt("v1", "base")
    # the base state rules survive character for character, and lead
    assert STATE_RULES_BASE_QA.startswith(STATE_RULES)
    assert STATE_RULES in sp
    # what was ported
    assert "character for character" in sp        # quoting
    assert "trochlea" in sp and "traumatic bruise" in sp   # attribution
    # what was NOT ported: the strict state machinery that measured worse
    for banned in ("STEP 1", "STEP 2", "STEP 3", "STEP 4", "FIND THE STRUCTURE FIRST",
                   "the single most common mistake"):
        assert banned not in sp
    # and the one clause of the ported paragraphs that is itself a state rule is gone
    assert "the state is \"absent\" and evidence is" not in sp
    assert "keep the state you\ndecided from the report" in sp


def test_base_qa_does_not_disturb_base_or_strict():
    import hashlib

    from rsna_kad.labels.prompts import STATE_RULES

    assert hashlib.md5(system_prompt("v1", "strict").encode()).hexdigest() == "a036bd85a1265e03a7080e92f5a0878f"
    assert STATE_RULES in system_prompt("v1", "base")
    # v2 keeps the base checklist head: base_qa changes no state semantics
    assert user_prompt("R", "v2", "base_qa").count("Checklist - go through the labels") == 1
    fs = [{"StudyInstanceUID": "u", "report": "r", "output": {}}]
    a = build_messages("one", "en", "v1", fs, "base_qa")
    b = build_messages("two", "en", "v1", fs, "base_qa")
    assert a[:-1] == b[:-1]
