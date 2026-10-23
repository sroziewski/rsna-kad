"""Chat prompts for the LLM extractor: definitions with the host thresholds (research.md §1),
two variants (V1 definitions-first, V2 report-first + checklist) sharing the few-shots."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rsna_kad.constants import LABEL_KEYS

FEWSHOT_DIR = Path(__file__).resolve().parents[3] / "data" / "fewshot"
FEWSHOT_FALLBACK_LANG = "en"
VARIANTS: tuple[str, ...] = ("v1", "v2")
# Rule wording used inside a variant. "base" is the stage-5 text; "strict" rewrites the same
# semantics against the four instruction-following failures measured in qa/stage5/report.md §5.
PROMPT_RULES: tuple[str, ...] = ("base", "strict", "strict2", "base_qa")

# Thresholds quoted verbatim from the host statements (research.md §1).
THRESHOLDS: dict[str, str] = {
    "acl": "\">50% of fibres disrupted\" or discontinuity (mild signal change = negative)",
    "mcl": "high-grade partial or complete acute tear (low-grade sprain = negative)",
    "med_men": "signal \"definitely contacting the meniscal surface on >=2 images\" (intrasubstance degeneration = negative)",
    "lat_men": "signal \"definitely contacting the meniscal surface on >=2 images\" (intrasubstance degeneration = negative)",
    "med_oa": "cartilage loss >50% thickness over >=~1 cm",
    "lat_oa": "cartilage loss >50% thickness over >=~1 cm",
    "pf_oa": "cartilage loss >50% thickness over >=~1 cm",
    "effusion": "moderate or large only (small/trace effusion = mild)",
    "baker": "moderate or large only (small cyst = mild)",
    "synovitis": "synovial thickening, proliferation or enhancement described as synovitis (effusion alone is not synovitis)",
    "contusion": "traumatic bone marrow oedema / bone bruise (subchondral oedema of arthritis or insufficiency alone does not count)",
    "fracture": "any fracture line, cortical break, avulsion, depression or subchondral/insufficiency fracture of femur, tibia, fibula or patella",
}

DEFINITIONS: dict[str, str] = {
    "acl": "anterior cruciate ligament tear",
    "mcl": "medial (tibial) collateral ligament tear",
    "med_men": "medial meniscus tear",
    "lat_men": "lateral meniscus tear",
    "med_oa": "medial tibiofemoral compartment osteoarthritis / cartilage loss",
    "lat_oa": "lateral tibiofemoral compartment osteoarthritis / cartilage loss",
    "pf_oa": "patellofemoral (patella or trochlea) osteoarthritis / cartilage loss",
    "effusion": "knee joint effusion",
    "synovitis": "synovitis",
    "baker": "Baker's (popliteal) cyst",
    "contusion": "bone contusion / bone bruise",
    "fracture": "fracture",
}

STATE_RULES = """For each of the 12 labels decide a STATE:
- "affirmed": the report states the finding is present, at any severity (including mild, partial, small or low-grade).
- "hedged": the report calls it possible, probable, suspected, questionable, cannot be excluded, or equivocal.
- "negated": the report explicitly states it is absent, normal, intact, preserved or unremarkable.
- "absent": the report does not mention the finding or the structure at all.
A normal statement about a group negates every member of the group ("cruciate and collateral ligaments intact" negates acl and mcl; "menisci normal" negates med_men and lat_men; "cartilage normal" negates med_oa, lat_oa and pf_oa; "no bone marrow oedema, no fracture" negates contusion and fracture). Degeneration, mucoid change, sprain or signal change without a tear is still "affirmed" for that label, with severity "mild".

SEVERITY (for affirmed or hedged; "none" otherwise):
- "mild": mild, minimal, small, trace, slight, low-grade, grade 1-2, superficial, partial-thickness, intrasubstance, mucoid, early, incipient.
- "moderate": moderate, medium.
- "severe": severe, extensive, large, complete, full-thickness, high-grade, grade 3-4, advanced, displaced, bucket-handle.
- "unspecified": the finding is affirmed but no grade or size is given.
Mild findings below the annotator threshold still get state "affirmed" with severity "mild" - never downgrade them to "negated"; the scoring decides.

EVIDENCE: a short verbatim quote copied from the report (max 120 characters, in the report's own language) that supports the state. Empty string "" for "absent"."""

_SEVERITY_RULES = """SEVERITY (for affirmed or hedged; "none" otherwise):
- "mild": mild, minimal, small, trace, slight, low-grade, grade 1-2, superficial, partial-thickness, intrasubstance, mucoid, early, incipient.
- "moderate": moderate, medium.
- "severe": severe, extensive, large, complete, full-thickness, high-grade, grade 3-4, advanced, displaced, bucket-handle.
- "unspecified": the finding is affirmed but no grade or size is given."""

# The four failure modes the stage-5 audit counted (24 state errors / 300 cells): "negated" written
# for structures the report never names (14), sub-threshold findings turned into "negated" (4),
# findings attributed to the wrong named structure (6), and quotes that are not in the report (6).
#
# `strict2` differs from `strict` only in the precedence between STEP 1 and STEP 3. Measured on the
# qa/llm_ab subset, `strict` made the model read STEP 1 ("a finding belongs to the structure named in
# the same sentence") as licence to leave the three OA labels "absent" whenever a normal statement
# named no compartment; on the gold cells where it disagreed with `base` it lost 11-0 on med_oa and
# 8-2 on pf_oa. `strict2` states that the group rule wins and names the unqualified OA negations.
def _strict_rules(group_precedence: bool) -> str:
    step1_tail = ("\nSTEP 3 overrides this: a normal statement that names no compartment and no side still "
                  "negates every member of its group. Use STEP 1 to place a *positive* finding, never to "
                  "throw a negation away.\n" if group_precedence else "\n")
    extra_group = ("- \"no degenerative change\", \"no osteoarthritis\", \"no joint space narrowing\", \"the joint "
                   "spaces are preserved\", \"no chondral abnormality\" name no compartment but are still group "
                   "statements -> negated for med_oa, lat_oa AND pf_oa;\n" if group_precedence else "")
    return """For each of the 12 labels decide a STATE. Work through the four steps in order.

STEP 1 - FIND THE STRUCTURE FIRST. Before you choose a state, look for the sentence that names this
label's own structure. A""" + (" positive" if group_precedence else "") + """ finding belongs only to the structure named in the same sentence:
- cartilage damage at the trochlea, the trochlear groove or behind the patella is pf_oa, never med_oa or lat_oa;
- med_oa is the medial femorotibial compartment only, lat_oa the lateral femorotibial compartment only;
- subchondral, reactive or degenerative marrow oedema in an arthritic or overloaded joint is not contusion;
  contusion needs a traumatic bruise / traumatic marrow oedema;
- a tibial-plateau, condylar or capsular finding is not a meniscal finding;
- a ganglion, popliteal vessel or bursa finding is not baker unless the report calls it a popliteal / Baker's cyst.
If no sentence names this structure and no sentence names this finding, the state is "absent".""" + step1_tail + """
STEP 2 - CHOOSE THE STATE.
- "absent": the report never mentions this finding and never mentions its structure, not even inside a
  group phrase. Most reports list only what was seen, so "absent" is the right answer for most labels.
  Never write "negated" for a structure the report does not name - that is the single most common mistake.
- "negated": the report itself says this finding is not there, or calls this structure normal, intact,
  preserved, continuous or unremarkable. You must be able to point at the words that say so.
- "hedged": possible, probable, suspected, questionable, equivocal, cannot be excluded.
- "affirmed": the report states the finding is present, at any severity.

STEP 3 - GROUP AND PLURAL STATEMENTS. A normal statement about a group negates every member of that
group, and only the members of that group:
- "the menisci are normal", "both menisci are intact", "no meniscal tear" -> negated for BOTH med_men and lat_men;
- "the cruciate and collateral ligaments are intact" -> negated for acl AND mcl;
- "the cartilage is normal", "no chondral loss", "no arthrosis" -> negated for med_oa, lat_oa AND pf_oa;
""" + extra_group + """- "no bone marrow oedema", "the bones are normal", "osseous structures unremarkable" -> negated for contusion AND fracture;
- "no internal derangement" covers acl, mcl, med_men and lat_men only; it says nothing about the OA
  labels, effusion, synovitis, baker, contusion or fracture, which stay "absent" unless named elsewhere;
- "no joint effusion" negates effusion only - not synovitis and not baker;
- a statement about one side only ("the medial meniscus is intact") negates that side only; the other
  side stays "absent" unless it is mentioned too.

STEP 4 - SUB-THRESHOLD FINDINGS ARE POSITIVE, NEVER NEGATED. Meniscal degeneration, mucoid or myxoid
change, intrasubstance signal that does not reach the meniscal surface, ligament sprain, partial or
low-grade tear, signal change without discontinuity, chondromalacia, chondropathy, trace effusion,
small cyst: state "affirmed" with severity "mild". The report describes a real finding; the annotator
threshold is applied afterwards by our scoring, not by you. A meniscus that shows degeneration is not
a normal meniscus, so it is never "negated"; writing "negated" or "absent" for it loses the finding.

""" + _SEVERITY_RULES + """
Mild findings below the annotator threshold still get state "affirmed" with severity "mild" - never downgrade them to "negated"; the scoring decides.

EVIDENCE: copy a span of the report character for character - the same words, accents and spelling, in
the report's own language, at most 120 characters. It must occur in the report exactly as you write it.
Do not translate it, do not correct or tidy the wording, do not join parts of two different sentences,
do not write a sentence that is not there. The span you copy must itself mention this label's own
structure or finding""" + (", or the group phrase that covers it" if group_precedence else " (STEP 1)") + """. If you cannot copy such a span, the state is "absent" and evidence is ""."""


STATE_RULES_STRICT = _strict_rules(False)
STATE_RULES_STRICT2 = _strict_rules(True)

# The single carry-forward the qa/llm_ab A/B licensed for targets v1.1: `strict` cut quote misses
# by 36 % and misattribution by 18 %, but its STEP 1/STEP 2 *state* rules lost significantly on
# gold (p=0.03/0.02) because they turn confident negations into "absent", and the host graded
# borderline findings negative. `base_qa` therefore keeps STATE_RULES byte for byte and adds only
# the two paragraphs that never touch the state: where a finding belongs (attribution) and how to
# copy the quote (quoting).
#
# Two deliberate edits to the ported text. `strict`'s attribution block is headed "before you
# choose a state" and ends "then the state is absent"; that sentence IS the regression, so it is
# left behind and the block is demoted to a tie-breaker between labels. `strict`'s quoting
# paragraph ends "if you cannot copy such a span, the state is absent" - the same mechanism, so
# the fallback here keeps the state and drops only the quote.
ATTRIBUTION_RULES = """ATTRIBUTION - which label a finding belongs to. This decides *which* label a
finding is filed under; it never changes whether the finding is there:
- cartilage damage at the trochlea, the trochlear groove or behind the patella is pf_oa, never med_oa or lat_oa;
- med_oa is the medial femorotibial compartment only, lat_oa the lateral femorotibial compartment only;
- subchondral, reactive or degenerative marrow oedema in an arthritic or overloaded joint is not contusion;
  contusion needs a traumatic bruise / traumatic marrow oedema;
- a tibial-plateau, condylar or capsular finding is not a meniscal finding;
- a ganglion, popliteal vessel or bursa finding is not baker unless the report calls it a popliteal / Baker's cyst.
A finding that names one structure is not evidence about a different structure."""

QUOTING_RULES = """QUOTING - copy the span of the report character for character: the same words,
accents and spelling, in the report's own language, at most 120 characters. It must occur in the
report exactly as you write it. Do not translate it, do not correct or tidy the wording, do not
join parts of two different sentences, do not write a sentence that is not there. The span you
copy must itself mention this label's own structure or finding, or the group phrase that covers
it. If no such span can be copied word for word, leave the evidence empty - keep the state you
decided from the report, and never invent a quote to justify it."""

STATE_RULES_BASE_QA = "\n\n".join([STATE_RULES, ATTRIBUTION_RULES, QUOTING_RULES])

EXTRA_FIELDS = """Also report:
- "knee_side": "left" or "right" when the report says which knee was examined, otherwise "unknown".
- "postoperative": true when the report describes prior knee surgery (ACL graft/reconstruction, meniscectomy, arthroscopy, hardware).
- "normal_template": true when the report is a normal-study template with no abnormal finding at all."""

OUTPUT_FORMAT = ("Answer with exactly one JSON object and nothing else:\n"
                 "{\"knee_side\": \"left|right|unknown\", \"postoperative\": bool, \"normal_template\": bool, "
                 "\"labels\": {" + ", ".join(f"\"{k}\": {{\"state\": ..., \"severity\": ..., \"evidence\": ...}}" for k in LABEL_KEYS) + "}}")


def _label_block() -> str:
    lines = ["Labels, with the thresholds the image annotators used (a finding at or above the threshold "
             "is a definite positive; below it, still mark affirmed with severity mild):"]
    for key in LABEL_KEYS:
        lines.append(f"- {key}: {DEFINITIONS[key]}. Threshold: {THRESHOLDS[key]}.")
    return "\n".join(lines)


def _state_rules(rules: str) -> str:
    if rules == "base":
        return STATE_RULES
    if rules == "strict":
        return STATE_RULES_STRICT
    if rules == "strict2":
        return STATE_RULES_STRICT2
    if rules == "base_qa":
        return STATE_RULES_BASE_QA
    raise ValueError(f"unknown prompt rules {rules!r}")


def system_prompt(variant: str, rules: str = "base") -> str:
    role = ("You are a musculoskeletal radiologist turning a knee MRI report into structured findings. "
            "The report may be written in English, Spanish, Turkish, Croatian, Greek, German, Bulgarian, Dutch or French; "
            "read it in its own language and answer only with JSON.")
    if variant == "v1":
        return "\n\n".join([role, _state_rules(rules), _label_block(), EXTRA_FIELDS, OUTPUT_FORMAT])
    if variant == "v2":
        _state_rules(rules)  # reject an unknown rule set on both variants
        return "\n\n".join([role, "You will receive the report first, then a checklist to work through. "
                            "Base every answer strictly on the report text.", OUTPUT_FORMAT])
    raise ValueError(f"unknown prompt variant {variant!r}")


def _checklist(rules: str = "base") -> str:
    head = ("Checklist - go through the labels in this order and, for each, ask: (1) is the structure or finding "
            "mentioned at all (no -> absent)? (2) is it explicitly normal/absent (-> negated)? (3) is it uncertain "
            "(-> hedged)? (4) otherwise affirmed - then grade the severity from the report's wording and copy a short "
            "verbatim quote as evidence.")
    if rules in ("strict", "strict2"):
        head = ("Checklist - for each label in this order: (1) name the sentence that mentions this label's own "
                "structure; if there is none the state is \"absent\" (never \"negated\"); (2) does that sentence call "
                "it normal/intact (-> negated), uncertain (-> hedged) or present (-> affirmed, including "
                "degeneration and other sub-threshold findings, severity mild)? (3) copy an exact span of that "
                "sentence as evidence.")
    lines = [head, ""]
    for key in LABEL_KEYS:
        lines.append(f"- {key}: {DEFINITIONS[key]}; annotator threshold: {THRESHOLDS[key]}.")
    lines += ["", _state_rules(rules), "", EXTRA_FIELDS]
    return "\n".join(lines)


def user_prompt(report: str, variant: str, rules: str = "base") -> str:
    report = report.strip()
    if variant == "v1":
        _state_rules(rules)
        return f"Report:\n\"\"\"\n{report}\n\"\"\"\n\nReturn the JSON."
    if variant == "v2":
        return f"Report:\n\"\"\"\n{report}\n\"\"\"\n\n{_checklist(rules)}\n\nReturn the JSON."
    raise ValueError(f"unknown prompt variant {variant!r}")


def load_fewshot(lang: str, fewshot_dir: str | Path | None = None) -> list[dict[str, Any]]:
    """Few-shot examples for a language (English fallback); each has StudyInstanceUID, report, output."""
    d = Path(fewshot_dir) if fewshot_dir else FEWSHOT_DIR
    path = d / f"{lang}.json"
    if not path.exists():
        path = d / f"{FEWSHOT_FALLBACK_LANG}.json"
    if not path.exists():
        return []
    examples = json.loads(path.read_text(encoding="utf-8"))
    for ex in examples:
        for field in ("StudyInstanceUID", "report", "output"):
            if field not in ex:
                raise ValueError(f"{path}: few-shot example lacks {field!r}")
    return examples


def assert_no_gold(fewshot_ids: set[str] | list[str], gold_ids: set[str] | list[str]) -> None:
    """Gold-58 rows are the validation set: a few-shot example drawn from them is a leak."""
    overlap = set(fewshot_ids) & set(gold_ids)
    if overlap:
        raise AssertionError(f"few-shot examples contain gold StudyInstanceUIDs: {sorted(overlap)[:5]}")


def build_messages(report: str, lang: str, variant: str,
                   fewshot: list[dict[str, Any]] | None = None,
                   rules: str = "base") -> list[dict[str, str]]:
    """System + (user, assistant) x few-shots + user. Few-shots go first so the prefix cache
    covers everything but the report."""
    if fewshot is None:
        fewshot = load_fewshot(lang)
    messages = [{"role": "system", "content": system_prompt(variant, rules)}]
    for ex in fewshot:
        messages.append({"role": "user", "content": user_prompt(ex["report"], variant, rules)})
        messages.append({"role": "assistant", "content": json.dumps(ex["output"], ensure_ascii=False)})
    messages.append({"role": "user", "content": user_prompt(report, variant, rules)})
    return messages
