import json

import pandas as pd
import pytest

from rsna_kad.constants import LABEL_KEYS, SEVERITIES, STATES
from rsna_kad.labels.schema import (JSON_SCHEMA, LABEL_TABLE_COLUMNS, ReportLabels, finalize_label_table,
                                    label_row, parse_output, validate_label_table, write_label_table)


def _doc(**over):
    labels = {k: {"state": "absent", "severity": "none", "evidence": ""} for k in LABEL_KEYS}
    labels["acl"] = {"state": "affirmed", "severity": "severe", "evidence": "complete ACL tear"}
    labels["effusion"] = {"state": "negated", "severity": "none", "evidence": "no effusion"}
    doc = {"knee_side": "left", "postoperative": False, "normal_template": False, "labels": labels}
    doc.update(over)
    return doc


def test_schema_covers_all_keys_and_enums():
    props = JSON_SCHEMA["properties"]["labels"]["properties"]
    assert tuple(props) == LABEL_KEYS
    assert JSON_SCHEMA["properties"]["labels"]["required"] == list(LABEL_KEYS)
    assert props["acl"]["properties"]["state"]["enum"] == list(STATES)
    assert props["acl"]["properties"]["severity"]["enum"] == list(SEVERITIES)
    assert JSON_SCHEMA["additionalProperties"] is False


def test_parse_roundtrip():
    rec = parse_output(json.dumps(_doc()))
    assert rec is not None
    assert rec.knee_side == "left"
    assert rec.cell("acl").state == "affirmed" and rec.cell("acl").severity == "severe"
    assert rec.cell("effusion").state == "negated" and rec.cell("effusion").severity == "none"
    assert rec.cell("baker").state == "absent"
    assert json.loads(rec.to_json())["labels"]["acl"]["evidence"] == "complete ACL tear"


def test_parse_tolerates_think_and_fences():
    text = "<think>\nreasoning\n</think>\n```json\n" + json.dumps(_doc()) + "\n```"
    assert parse_output(text) is not None


def test_parse_rejects_bad_enum_missing_key_and_garbage():
    bad = _doc()
    bad["labels"]["acl"]["state"] = "present"
    assert parse_output(json.dumps(bad)) is None
    missing = _doc()
    del missing["labels"]["fracture"]
    assert parse_output(json.dumps(missing)) is None
    assert parse_output("not json") is None
    assert parse_output("") is None
    assert parse_output(None) is None


def test_parse_normalises_severity_and_evidence():
    doc = _doc()
    doc["labels"]["mcl"] = {"state": "affirmed", "severity": "none", "evidence": "x" * 300}
    doc["labels"]["baker"] = {"state": "absent", "severity": "mild", "evidence": "should vanish"}
    rec = parse_output(json.dumps(doc))
    assert rec.cell("mcl").severity == "unspecified"
    assert len(rec.cell("mcl").evidence) == 120
    assert rec.cell("baker").severity == "none" and rec.cell("baker").evidence == ""


def test_label_row_and_table(tmp_path):
    rows = [label_row("s1", "en", "rules", "v0", parse_output(json.dumps(_doc()))),
            label_row("s2", "es", "rules", "v0", None)]
    df = pd.DataFrame(rows)
    assert list(df.columns) == list(LABEL_TABLE_COLUMNS)
    assert not df.loc[1, "parse_ok"] and df.loc[1, "acl_state"] == "absent"
    validate_label_table(df, n_rows=2)
    out = tmp_path / "t.parquet"
    write_label_table(df, out)
    back = pd.read_parquet(out)
    assert str(back["acl_state"].dtype) == "category"
    assert out.with_suffix(".csv").exists()
    with pytest.raises(ValueError):
        validate_label_table(pd.concat([df, df]), n_rows=4)  # duplicate ids


def test_finalize_keeps_extra_columns():
    df = pd.DataFrame([label_row("s1", "en", "r", "v", ReportLabels())])
    df["acl_mentioned"] = True
    assert list(finalize_label_table(df).columns)[-1] == "acl_mentioned"
