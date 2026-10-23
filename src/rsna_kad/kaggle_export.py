"""Kaggle packaging (architecture §2.7/§2.8/§3.2): the weights dataset directory
(models + ensemble.json + wheels + VERSION), the generated self-contained inference script
(modules inlined between `# --- inline:` markers) and thin wrappers around the kaggle CLI."""
from __future__ import annotations

import ast
import hashlib
import importlib
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from rsna_kad.constants import ID_COL, LABEL_KEYS, LABELS
from rsna_kad.infer import ENSEMBLE_NAME, META_SUFFIX, WEIGHTS_SUFFIX, load_member
from rsna_kad.model import BUNDLE_NAME, WEIGHTS_NAME

DATASET_SLUG = "rsna-kad-weights"
KERNEL_SLUG = "rsna-kad-mvp"
COMPETITION = "rsna-knee-abnormality-detection"
MODELS_DIR = "models"
WHEELS_DIR = "wheels"
VERSION_NAME = "VERSION"
DATASET_META = "dataset-metadata.json"
KERNEL_META = "kernel-metadata.json"
NOTEBOOK_NAME = "kaggle_inference.py"
WHEEL_PACKAGES = ("pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg", "pylibjpeg-rle", "python-gdcm")
WHEEL_PYTHONS = ("3.12", "3.11")
WHEEL_PLATFORMS = ("manylinux2014_x86_64", "manylinux_2_28_x86_64")

# Modules inlined into the notebook, in dependency order; None = every top-level
# definition, a tuple = only those names (their module-level imports are still collected).
INLINE_MODULES: tuple[tuple[str, tuple[str, ...] | None], ...] = (
    ("constants", None),
    ("config", ("T", "CacheConfig", "FoldsConfig", "TargetsConfig", "ModelConfig", "TrainConfig", "InferConfig",
                "from_dict", "_wants_tuple", "to_dict", "params_sha256")),
    ("dicom", None),
    ("series", None),
    ("preprocess", None),
    ("laterality", None),
    ("data", ("sample_centres", "triplet_indices", "read_triplets", "warp_slices", "view_params",
              "view_mm_for")),
    ("model", None),
    ("infer", None),
)
ALLOWED_PACKAGES = frozenset({"numpy", "pandas", "torch", "timm", "pydicom", "cv2", "safetensors"})
# Third-party imports of inlined modules whose users are not inlined.
DROPPED_IMPORTS = frozenset({"yaml"})
MARKER = "# --- inline:"


# ------------------------------------------------------------------ bundle export


def member_name(meta: Mapping[str, Any]) -> str:
    if meta["fold"] == "full":
        return f"{meta['run_id']}_full"
    return f"{meta['run_id']}_f{int(meta['fold'])}"


def prior_from_oof(oof_parquet: str | os.PathLike[str]) -> list[float]:
    """Train-mean prediction per label: the row a study gets when nothing can be inferred."""
    oof = pd.read_parquet(oof_parquet)
    return [float(oof[f"{k}_pred"].mean()) for k in LABEL_KEYS]


def copy_wheels(wheels_dir: str | os.PathLike[str], out_dir: Path) -> list[str]:
    out = out_dir / WHEELS_DIR
    out.mkdir(parents=True, exist_ok=True)
    names = []
    for whl in sorted(Path(wheels_dir).glob("*.whl")):
        shutil.copy2(whl, out / whl.name)
        names.append(whl.name)
    return names


def download_wheels(out_dir: str | os.PathLike[str], python: str = sys.executable,
                    pythons: Sequence[str] = WHEEL_PYTHONS, packages: Sequence[str] = WHEEL_PACKAGES) -> list[str]:
    """pip download of the manylinux decoder wheels for each CPython version (Kaggle's
    interpreter is not pinned by us, so both 3.12 and 3.11 ship)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for py in pythons:
        cmd = [python, "-m", "pip", "download", "--no-cache-dir", "--disable-pip-version-check", "--no-deps",
               "--only-binary=:all:", "--python-version", py, "--implementation", "cp",
               "--abi", "cp" + py.replace(".", ""), "-d", str(out)]
        for plat in WHEEL_PLATFORMS:
            cmd += ["--platform", plat]
        subprocess.run(cmd + list(packages), check=True)
    return sorted(p.name for p in out.glob("*.whl"))


def export_bundle(run_dirs: Sequence[str | os.PathLike[str]], out_dir: str | os.PathLike[str], mode: str = "main",
                  wheels_dir: str | os.PathLike[str] | None = None, tta: bool | None = None,
                  workers: int = 3, wall_guard_hours: float = 8.25, prior: Sequence[float] | None = None,
                  user: str = "", verify: bool = True, k_infer: int | None = None,
                  git_sha: str = "") -> dict[str, Any]:
    """runs/<run_id>/fold{k} dirs -> <out_dir>/{models/<arm>_f<k>.{safetensors,json}, ensemble.json,
    wheels/, VERSION, dataset-metadata.json}. tta None = main → z-offset TTA, efficiency → none.

    git_sha (the exporting tree's commit) goes into ensemble.json and into every member json
    whose training left `git_sha` empty; a sha recorded at training time is kept, since it
    describes the code that produced the weights.

    k_infer overrides the triplet count of every member. `Member.spec` reads it from the member
    json, not from ensemble.json, so the override has to be written into each copied bundle -
    ensemble.json's own K_infer is a record of what was shipped."""
    if mode not in ("main", "efficiency"):
        raise ValueError(f"mode must be main|efficiency, got {mode!r}")
    if not run_dirs:
        raise ValueError("export_bundle: no run dirs")
    out = Path(out_dir)
    (out / MODELS_DIR).mkdir(parents=True, exist_ok=True)
    tta = (mode == "main") if tta is None else tta
    members, names = [], []
    for rd in run_dirs:
        rd = Path(rd)
        meta = json.loads((rd / BUNDLE_NAME).read_text(encoding="utf-8"))
        if not meta.get("git_sha"):
            meta["git_sha"] = git_sha
        if k_infer is not None:
            meta["arch"]["K_infer"] = int(k_infer)
            meta["config"]["model"]["k_infer"] = int(k_infer)
        name = member_name(meta)
        if name in names:
            raise ValueError(f"duplicate member {name} from {rd}")
        names.append(name)
        prefix = out / MODELS_DIR / name
        shutil.copy2(rd / WEIGHTS_NAME, str(prefix) + WEIGHTS_SUFFIX)
        (Path(str(prefix) + META_SUFFIX)).write_text(json.dumps(meta, indent=1, sort_keys=True), encoding="utf-8")
        if verify:
            # views=True: the notebook builds this member's inputs through infer.study_inputs.
            load_member(prefix, "cpu", meta, views=True)
        members.append({"path": f"{MODELS_DIR}/{name}", "weight": 1.0, "tta": bool(tta),
                        "backbone": meta["arch"]["backbone"], "img_px": meta["arch"]["img_px"],
                        "metrics": {k: meta.get("metrics", {}).get(k) for k in ("oof_weak_auc", "gold_auc", "epoch")}})
    if prior is None:
        oofs = [Path(rd) / "oof.parquet" for rd in run_dirs if (Path(rd) / "oof.parquet").exists()]
        prior = (list(np.mean([prior_from_oof(p) for p in oofs], axis=0)) if oofs
                 else [0.5] * len(LABELS))
    if k_infer is None:
        k_infer = int(json.loads((Path(run_dirs[0]) / BUNDLE_NAME).read_text(encoding="utf-8"))["arch"]["K_infer"])
    wheels = copy_wheels(wheels_dir, out) if wheels_dir else sorted(p.name for p in (out / WHEELS_DIR).glob("*.whl"))
    version = f"{out.name} {time.strftime('%Y-%m-%dT%H:%M:%S%z')} " + " ".join(names)
    doc = {
        "mode": mode, "members": members, "K_infer": k_infer, "workers": int(workers),
        "wall_guard_hours": float(wall_guard_hours), "prior": [float(p) for p in prior], "labels": list(LABELS),
        "wheels": wheels, "version": version, "git_sha": git_sha,
    }
    (out / ENSEMBLE_NAME).write_text(json.dumps(doc, indent=1), encoding="utf-8")
    (out / VERSION_NAME).write_text(version + "\n", encoding="utf-8")
    write_dataset_metadata(out, user)
    return doc


def write_dataset_metadata(out_dir: str | os.PathLike[str], user: str, slug: str = DATASET_SLUG) -> Path:
    path = Path(out_dir) / DATASET_META
    doc = {"title": slug, "id": f"{user}/{slug}" if user else slug, "licenses": [{"name": "CC0-1.0"}]}
    path.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    return path


def kernel_metadata(user: str, slug: str = KERNEL_SLUG, dataset_slug: str = DATASET_SLUG,
                    code_file: str = f"../{NOTEBOOK_NAME}") -> dict[str, Any]:
    return {
        "id": f"{user}/{slug}", "title": slug, "code_file": code_file, "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_tpu": False,
        "enable_internet": False, "dataset_sources": [f"{user}/{dataset_slug}"],
        "competition_sources": [COMPETITION], "kernel_sources": [], "model_sources": [],
    }


def write_kernel_metadata(kernel_dir: str | os.PathLike[str], user: str, **kw: Any) -> Path:
    path = Path(kernel_dir) / KERNEL_META
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(kernel_metadata(user, **kw), indent=1) + "\n", encoding="utf-8")
    return path


# ------------------------------------------------------------------ notebook generation


def _module_source(name: str) -> tuple[str, ast.Module]:
    mod = importlib.import_module(f"rsna_kad.{name}")
    src = inspect.getsource(mod)
    return src, ast.parse(src)


def _import_root(node: ast.Import | ast.ImportFrom) -> str:
    if isinstance(node, ast.ImportFrom):
        return (node.module or "").split(".")[0]
    return node.names[0].name.split(".")[0]


def _node_names(node: ast.stmt) -> list[str]:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return [node.name]
    if isinstance(node, ast.Assign):
        return [t.id for t in node.targets if isinstance(t, ast.Name)]
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return [node.target.id]
    return []


def _segment(src: str, node: ast.stmt) -> str:
    lines = src.splitlines(keepends=True)
    start = node.lineno - 1
    if getattr(node, "decorator_list", None):
        start = min(d.lineno for d in node.decorator_list) - 1
    return "".join(lines[start:node.end_lineno])


def inline_module(name: str, only: tuple[str, ...] | None) -> tuple[list[str], str]:
    """(third-party/stdlib import lines, body) of one module with rsna_kad imports removed."""
    src, tree = _module_source(name)
    stdlib = set(sys.stdlib_module_names)
    imports, body = [], []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            root = _import_root(node)
            if root in ("rsna_kad", "__future__") or root in DROPPED_IMPORTS:
                continue
            if root not in stdlib and root not in ALLOWED_PACKAGES:
                raise ValueError(f"{name}: import of {root!r} is not allowed in the notebook")
            imports.append(_segment(src, node).rstrip("\n"))
            continue
        if isinstance(node, ast.Expr) and isinstance(getattr(node, "value", None), ast.Constant) \
                and isinstance(node.value.value, str):
            continue  # module docstring
        if only is not None:
            names = _node_names(node)
            if not names or not all(n in only for n in names):
                continue
        body.append(_segment(src, node).rstrip("\n"))
    return imports, "\n\n".join(body) + "\n"


PRELUDE = '''
def _find_wheels(weights: str) -> str:
    """<weights>/wheels, or the first wheels/ dir holding the decoder wheels under /kaggle/input
    (Kaggle mounts datasets at /kaggle/input/<slug> or /kaggle/input/datasets/<user>/<slug>)."""
    direct = os.path.join(weights, "{wheels}")
    if os.path.isdir(direct):
        return direct
    frontier = ["/kaggle/input"] if os.path.isdir("/kaggle/input") else []
    for _depth in range(4):
        nxt = []
        for d in frontier:
            try:
                names = sorted(os.listdir(d))
            except OSError:
                continue
            for n in names:
                p = os.path.join(d, n)
                if not os.path.isdir(p) or n in ("test_series", "train_series", "models"):
                    continue
                if n == "{wheels}" and any(f.endswith(".whl") for f in os.listdir(p)):
                    return p
                nxt.append(p)
        frontier = nxt
    return direct


def _install_wheels() -> None:
    """Offline install of the DICOM decoder wheels shipped in the weights dataset
    (architecture §2.7); runs before pydicom is imported so its plugins are found."""
    argv = sys.argv[1:]
    weights = argv[argv.index("--weights") + 1] if "--weights" in argv and argv.index("--weights") + 1 < len(argv) \\
        else "{weights}"
    on_kaggle = os.path.isdir("/kaggle/working")
    if not (on_kaggle or "--install-wheels" in argv):
        return
    wheels = _find_wheels(weights)
    if not os.path.isdir(wheels):
        print(f"WARNING: no wheels directory found for {{weights}}; uncompressed DICOM still decodes", flush=True)
        return
    cmd = [sys.executable, "-m", "pip", "install", "--no-index", "--no-cache-dir", "--disable-pip-version-check",
           "--find-links", wheels, *{packages}]
    print("installing decoder wheels:", " ".join(cmd), flush=True)
    rc = subprocess.call(cmd)
    if rc != 0:
        print(f"WARNING: wheel install failed (exit {{rc}}); uncompressed DICOM still decodes", flush=True)


if __name__ == "__main__":
    _install_wheels()
'''

MAIN_BLOCK = '''
if __name__ == "__main__":
    sys.exit(notebook_main())
'''


def render_notebook() -> str:
    """The self-contained script text (deterministic for a given source tree)."""
    all_imports: list[str] = []
    sections = []
    for name, only in INLINE_MODULES:
        imports, body = inline_module(name, only)
        for line in imports:
            if line not in all_imports:
                all_imports.append(line)
        sections.append(f"{MARKER} {name}\n\n{body}")
    plain = sorted(l for l in all_imports if l.startswith("import "))
    froms = sorted(l for l in all_imports if l.startswith("from "))
    digest = hashlib.sha256("".join(sections).encode("utf-8")).hexdigest()[:16]
    prelude = PRELUDE.format(weights="/kaggle/input/rsna-kad-weights", wheels=WHEELS_DIR,
                             packages=repr(list(WHEEL_PACKAGES)))
    head = (
        "#!/usr/bin/env python3\n"
        '"""RSNA Knee Abnormality Detection - Kaggle inference script.\n\n'
        "GENERATED by rsna_kad.kaggle_export.build_notebook - do not edit; edit src/rsna_kad and rebuild.\n"
        "Reads <weights>/ensemble.json, installs the decoder wheels offline, preprocesses every test\n"
        "study exactly like the training cache and writes submission.csv.\n\n"
        "  python notebooks/kaggle_inference.py --data-dir <dir> --weights <export dir> --out submission.csv\n"
        f'"""\n# source digest {digest}\n'
        "from __future__ import annotations\n\n"
        "import os\nimport subprocess\nimport sys\n"
        f"\n{MARKER} prelude\n{prelude}\n"
        f"{MARKER} imports\n\n" + "\n".join(plain) + "\n\n" + "\n".join(froms) + "\n\n\n"
    )
    return head + "\n\n".join(sections) + f"\n{MARKER} main\n{MAIN_BLOCK}"


def notebook_digests(text: str) -> tuple[str, str]:
    """(source digest declared in the header, digest of the inlined sections as they are now).
    Equal for any unedited build, however old; a hand edit or a truncated copy breaks it."""
    m = re.search(r"^# source digest ([0-9a-f]{16})$", text, re.MULTILINE)
    start = text.find(f"{MARKER} {INLINE_MODULES[0][0]}\n")
    end = text.rfind(f"\n{MARKER} main\n")
    if not m or start < 0 or end < start:
        return (m.group(1) if m else "", "")
    parts = text[start:end].split(f"\n\n{MARKER} ")
    sections = [parts[0]] + [f"{MARKER} {p}" for p in parts[1:]]
    return m.group(1), hashlib.sha256("".join(sections).encode("utf-8")).hexdigest()[:16]


def build_notebook(out_path: str | os.PathLike[str]) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    text = render_notebook()
    compile(text, str(out), "exec")
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    return out


def notebook_imports(text: str) -> set[str]:
    return {_import_root(n) for n in ast.parse(text).body if isinstance(n, (ast.Import, ast.ImportFrom))}


# ------------------------------------------------------------------ kaggle CLI wrappers


def kaggle_env(config_dir: str | os.PathLike[str] | None = None) -> dict[str, str]:
    """Environment for the kaggle CLI: token from <config_dir>/access_token (passed as a
    path, never as a value), nothing written under $HOME."""
    env = dict(os.environ)
    cfg = Path(config_dir or env.get("KAGGLE_CONFIG_DIR") or Path.cwd() / ".kaggle")
    env["KAGGLE_CONFIG_DIR"] = str(cfg)
    token = cfg / "access_token"
    if token.is_file() and not env.get("KAGGLE_API_TOKEN"):
        env["KAGGLE_API_TOKEN"] = str(token)
    env.setdefault("PIP_NO_CACHE_DIR", "1")
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    return env


def kaggle_executable() -> str:
    beside = Path(sys.executable).with_name("kaggle" + (".exe" if os.name == "nt" else ""))
    if beside.exists():
        return str(beside)
    found = shutil.which("kaggle")
    if not found:
        raise FileNotFoundError("kaggle CLI not found: pip install --no-cache-dir kaggle")
    return found


def run_kaggle(args: Sequence[str], config_dir: str | os.PathLike[str] | None = None,
               check: bool = True) -> subprocess.CompletedProcess:
    cmd = [kaggle_executable(), *args]
    return subprocess.run(cmd, env=kaggle_env(config_dir), check=check, text=True, capture_output=True)


def push_dataset(bundle_dir: str | os.PathLike[str], message: str = "", new: bool = False,
                 config_dir: str | os.PathLike[str] | None = None) -> str:
    """`kaggle datasets create` (first version) or `version` (all later ones) of the export dir."""
    if new:
        args = ["datasets", "create", "-p", str(bundle_dir), "--dir-mode", "zip"]
    else:
        args = ["datasets", "version", "-p", str(bundle_dir), "--dir-mode", "zip", "-m", message or "update"]
    res = run_kaggle(args, config_dir)
    return res.stdout + res.stderr


def push_kernel(kernel_dir: str | os.PathLike[str], config_dir: str | os.PathLike[str] | None = None) -> str:
    res = run_kaggle(["kernels", "push", "-p", str(kernel_dir)], config_dir)
    return res.stdout + res.stderr


def kernel_status(kernel_id: str, config_dir: str | os.PathLike[str] | None = None) -> str:
    res = run_kaggle(["kernels", "status", kernel_id], config_dir)
    return res.stdout + res.stderr


def kernel_output(kernel_id: str, out_dir: str | os.PathLike[str],
                  config_dir: str | os.PathLike[str] | None = None) -> str:
    res = run_kaggle(["kernels", "output", kernel_id, "-p", str(out_dir)], config_dir)
    return res.stdout + res.stderr


def submit_kernel(kernel_id: str, version: int, message: str, competition: str = COMPETITION,
                  config_dir: str | os.PathLike[str] | None = None) -> str:
    """Code-competition submission of a finished kernel version's submission.csv."""
    res = run_kaggle(["competitions", "submit", competition, "-k", kernel_id, "-v", str(version),
                      "-f", "submission.csv", "-m", message], config_dir)
    return res.stdout + res.stderr


def submission_check(csv_path: str | os.PathLike[str]) -> dict[str, Any]:
    """Shape summary of a submission.csv (for reports/tests)."""
    df = pd.read_csv(csv_path)
    vals = df[list(LABELS)].to_numpy(dtype=np.float64)
    return {"rows": int(len(df)), "columns": list(df.columns), "ids_unique": bool(df[ID_COL].is_unique),
            "min": float(vals.min()), "max": float(vals.max()), "nan": int(np.isnan(vals).sum())}
