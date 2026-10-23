"""Stage 4 packaging (architecture §2.7/§2.8/§3.2): export layout, ensemble.json, the
generated notebook (reproducible, allowed imports only, runs end to end), kernel metadata,
kaggle CLI environment."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import STUDY_B, SyntheticData
from rsna_kad import kaggle_export as ke
from rsna_kad.constants import ID_COL, LABELS
from test_infer import make_tiny_export

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "notebooks" / ke.NOTEBOOK_NAME


@pytest.fixture(scope="module")
def tiny_export(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return make_tiny_export(tmp_path_factory.mktemp("export"))


def test_export_layout_and_ensemble_json(tiny_export: Path):
    doc = json.loads((tiny_export / "ensemble.json").read_text())
    assert doc["mode"] == "main" and doc["K_infer"] == 4 and doc["workers"] == 1
    assert [m["path"] for m in doc["members"]] == ["models/tiny_f0", "models/tiny_f1"]
    assert all(m["weight"] == 1.0 and m["tta"] is True for m in doc["members"])
    assert doc["labels"] == list(LABELS) and len(doc["prior"]) == 12
    # prior = mean over folds of the OOF prediction means (0.05*(j+1) + 0.01*k)
    assert np.allclose(doc["prior"], [0.05 * (j + 1) + 0.005 for j in range(12)])
    for m in doc["members"]:
        assert (tiny_export / (m["path"] + ".safetensors")).exists()
        meta = json.loads((tiny_export / (m["path"] + ".json")).read_text())
        assert meta["fingerprint"] and meta["arch"]["slots"] == ["sag_fs", "cor_fs", "ax_fs"]
    version = (tiny_export / "VERSION").read_text()
    assert version.startswith("tiny ") and "tiny_f1" in version and doc["version"] == version.strip()
    ds = json.loads((tiny_export / "dataset-metadata.json").read_text())
    assert ds["id"] == "someone/rsna-kad-weights" and ds["title"] == "rsna-kad-weights"
    with pytest.raises(ValueError, match="mode"):
        ke.export_bundle([tiny_export.parent.parent / "runs" / "tiny" / "fold0"], tiny_export, mode="fast")


def test_export_efficiency_mode_no_tta(tmp_path: Path):
    src = make_tiny_export(tmp_path, n_folds=1, with_oof=False)
    run_dir = tmp_path / "runs" / "tiny" / "fold0"
    doc = ke.export_bundle([run_dir], tmp_path / "eff", mode="efficiency", user="u")
    assert doc["members"][0]["tta"] is False and doc["prior"] == [0.5] * 12
    assert (tmp_path / "eff" / "models" / "tiny_f0.safetensors").stat().st_size == \
        (src / "models" / "tiny_f0.safetensors").stat().st_size


def test_kernel_metadata(tmp_path: Path):
    path = ke.write_kernel_metadata(tmp_path / "kernel", "someone")
    doc = json.loads(path.read_text())
    assert doc["id"] == "someone/rsna-kad-mvp" and doc["code_file"] == "../kaggle_inference.py"
    assert doc["kernel_type"] == "script" and doc["enable_gpu"] and not doc["enable_internet"]
    assert doc["dataset_sources"] == ["someone/rsna-kad-weights"]
    assert doc["competition_sources"] == ["rsna-knee-abnormality-detection"]


def test_committed_kernel_metadata_matches_generator():
    committed = json.loads((REPO / "notebooks" / "kernel" / "kernel-metadata.json").read_text())
    user = committed["id"].split("/")[0]
    assert committed == ke.kernel_metadata(user)


def test_kaggle_env_uses_token_path_and_project_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg = tmp_path / ".kaggle"
    cfg.mkdir()
    (cfg / "access_token").write_text("KGAT_secret")
    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
    env = ke.kaggle_env(cfg)
    assert env["KAGGLE_CONFIG_DIR"] == str(cfg) and env["KAGGLE_API_TOKEN"] == str(cfg / "access_token")
    assert "KGAT_secret" not in json.dumps(env)
    monkeypatch.setenv("KAGGLE_API_TOKEN", "already")
    assert ke.kaggle_env(cfg)["KAGGLE_API_TOKEN"] == "already"


# ------------------------------------------------------------------ notebook


def test_notebook_render_is_reproducible_and_self_contained():
    a, b = ke.render_notebook(), ke.render_notebook()
    assert a == b
    compile(a, "kaggle_inference.py", "exec")
    roots = ke.notebook_imports(a)
    assert "rsna_kad" not in roots and "yaml" not in roots
    assert roots <= set(sys.stdlib_module_names) | ke.ALLOWED_PACKAGES, roots - set(sys.stdlib_module_names)
    for name, _ in ke.INLINE_MODULES:
        assert f"{ke.MARKER} {name}\n" in a
    for needed in ("class StudyPreprocessor", "def predict_studies", "def write_submission", "def load_member",
                   "def series_to_volume", "def read_slice", "def assign_slots", "def sample_centres",
                   "class KneeMIL", "def fingerprint", "def notebook_main", "def _install_wheels"):
        assert needed in a, needed
    assert "def load_config" not in a and "import yaml" not in a
    assert a.index("from __future__ import annotations") < a.index("import numpy")


def test_committed_notebook_is_an_unedited_build():
    """Holds whatever src/ looks like now: the committed notebook is the kernel's code (kernel v10
    runs md5 e61dc63c, tag main_v3-shipped), so between releases it may be stale but never edited."""
    declared, actual = ke.notebook_digests(NOTEBOOK.read_text(encoding="utf-8"))
    assert declared and declared == actual
    fresh_declared, fresh_actual = ke.notebook_digests(ke.render_notebook())
    assert fresh_declared == fresh_actual
    tampered = NOTEBOOK.read_text(encoding="utf-8").replace("def assign_slots(", "def assign_slots_(", 1)
    assert ke.notebook_digests(tampered)[1] != declared


def test_committed_notebook_equals_fresh_build():
    assert NOTEBOOK.exists(), "run scripts/export_kaggle.py to generate the notebook"
    committed, fresh = NOTEBOOK.read_bytes(), ke.render_notebook().encode("utf-8")
    if committed != fresh and os.environ.get("RSNA_KAD_RELEASE") != "1":
        pytest.xfail("stale, not corrupt (see test_committed_notebook_is_an_unedited_build): src/ changed since "
                     "the shipped build; the next release rebuilds it. RSNA_KAD_RELEASE=1 makes this strict.")
    assert committed == fresh


def test_notebook_runs_locally_on_cpu(synthetic: SyntheticData, tiny_export: Path, tmp_path: Path):
    """The generated script, as a subprocess without rsna_kad on its path, produces a valid
    submission for the synthetic test split (fp32 CPU fallback of §4 row 4)."""
    nb = tmp_path / "kaggle_inference.py"
    ke.build_notebook(nb)
    out = tmp_path / "out" / "submission.csv"
    sample = synthetic.data_dir / "sample_submission.csv"
    pd.DataFrame({ID_COL: [STUDY_B], **{lab: [0.5] for lab in LABELS}}).to_csv(sample, index=False)
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    cmd = [sys.executable, str(nb), "--data-dir", str(synthetic.data_dir), "--weights", str(tiny_export),
           "--out", str(out), "--device", "cpu", "--workers", "1"]
    res = subprocess.run(cmd, cwd=str(tmp_path), env=env, text=True, capture_output=True, timeout=600)
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
    assert "installing decoder wheels" not in res.stdout  # only on Kaggle or with --install-wheels
    rep = ke.submission_check(out)
    assert rep["rows"] == 1 and rep["columns"] == [ID_COL, *LABELS] and rep["nan"] == 0
    assert 0.0 <= rep["min"] <= rep["max"] <= 1.0
    assert "0 prior rows" in res.stdout


def test_notebook_prelude_installs_only_when_asked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    text = ke.render_notebook()
    prelude = text[text.index(f"{ke.MARKER} prelude"):text.index(f"{ke.MARKER} imports")]
    ns: dict = {"os": os, "sys": sys, "subprocess": subprocess, "__name__": "not_main"}
    exec(prelude, ns)
    calls: list = []
    monkeypatch.setattr(subprocess, "call", lambda cmd: calls.append(cmd) or 0)
    wheels = tmp_path / "w" / "wheels"
    wheels.mkdir(parents=True)
    monkeypatch.setattr(sys, "argv", ["nb", "--weights", str(tmp_path / "w")])
    ns["_install_wheels"]()
    assert calls == []
    monkeypatch.setattr(sys, "argv", ["nb", "--weights", str(tmp_path / "w"), "--install-wheels"])
    ns["_install_wheels"]()
    assert len(calls) == 1 and "--no-index" in calls[0] and str(wheels) in calls[0]
    assert all(p in calls[0] for p in ke.WHEEL_PACKAGES)
