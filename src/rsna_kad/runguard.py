"""Guards for tools that read a training run (architecture §4.1: "never score a live chain",
and the export hazard).

train_fold rewrites fold<k>/{bundle.json,model.safetensors,oof.parquet,metrics.json} after every
epoch and deletes fold<k>/last.pt only after the final one, so a run directory looks complete
long before it is. Two independent checks, because each one alone has a blind spot:

- liveness: some process of this user other than the caller and its ancestors still names the
  run (its run_id, a config whose run_id it is, or a chain script mentioning either). The
  caller's ancestors are excluded so a chain may score or export its own finished folds.
- completeness: per fold, no last.pt, and the bundle holds the configured final epoch.

Not inlined into the notebook; scripts only."""
from __future__ import annotations

import fnmatch
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

SMOKE_PATTERNS = ("*_sweep_*", "*smoke*")
SCRIPT_SUFFIXES = (".sh", ".bash")
MAX_SCRIPT_BYTES = 256 * 1024
_CONFIG_RE = re.compile(r"(?<![\w./])[\w.][\w./-]*\.ya?ml\b")
_RUN_ID_RE = re.compile(r"^run_id:\s*['\"]?([\w.-]+)", re.MULTILINE)


class RunGuardError(RuntimeError):
    """A run is not safe to read (live) or to ship (incomplete, smoke, missing fold)."""


@dataclass(frozen=True)
class Proc:
    pid: int
    cmdline: str
    text: str  # cmdline + referenced chain scripts + "run_id=<id>" for every resolvable config
    run_ids: frozenset[str] = frozenset()
    open_paths: tuple[str, ...] = ()


def _ppid(pid: int, proc_root: str) -> int:
    try:
        stat = Path(proc_root, str(pid), "stat").read_text()
    except OSError:
        return 0
    # "pid (comm) state ppid ...": comm may contain spaces/parentheses, so split after the last ')'.
    return int(stat.rsplit(")", 1)[1].split()[1])


def ancestors(pid: int | None = None, proc_root: str = "/proc") -> set[int]:
    """pid and every ancestor up to init."""
    pid = os.getpid() if pid is None else pid
    out = set()
    while pid > 1 and pid not in out:
        out.add(pid)
        pid = _ppid(pid, proc_root)
    return out


def _resolve(token: str, cwd: Path | None) -> Path | None:
    p = Path(token)
    if not p.is_absolute():
        if cwd is None:
            return None
        p = cwd / p
    return p if p.is_file() else None


def _read_small(path: Path) -> str:
    try:
        with open(path, "rb") as fh:
            return fh.read(MAX_SCRIPT_BYTES).decode("utf-8", "replace")
    except OSError:
        return ""


def expand_text(cmdline: str, cwd: Path | None) -> str:
    """cmdline plus the text of every chain script it runs plus `run_id=<id>` for every config
    file named in either, so a run is matched even when only its config path is visible."""
    parts = [cmdline]
    for tok in cmdline.split():
        if tok.endswith(SCRIPT_SUFFIXES):
            script = _resolve(tok, cwd)
            if script is not None:
                parts.append(_read_small(script))
    text = "\n".join(parts)
    for name in sorted(set(_CONFIG_RE.findall(text))):
        cfg = _resolve(name, cwd)
        if cfg is not None:
            m = _RUN_ID_RE.search(_read_small(cfg))
            if m:
                parts.append(f"run_id={m.group(1)}")
    return "\n".join(parts)


WRITER_SCRIPTS = ("train.py", "predict_oof.py")
FULL = "full"


def seeded_run_id(run_id: str, seed: int | str) -> str:
    """run_id of `train.py --seed S` without --run-id: one run dir per seed of a recipe."""
    return f"{run_id}_s{seed}"


def _flag(argv: Sequence[str], name: str) -> str | None:
    for i, tok in enumerate(argv):
        if tok == name and i + 1 < len(argv):
            return argv[i + 1]
        if tok.startswith(name + "="):
            return tok.split("=", 1)[1]
    return None


def writer_run_ids(argv: Sequence[str], cwd: Path | None) -> frozenset[str]:
    """Run ids a `train.py` / `predict_oof.py` command line writes: --run-id, else the run_id of
    its --config, suffixed by --seed exactly as scripts/train.py does. Any other command (chain
    script, waiter, scorer) writes no run and gets none."""
    if not any(Path(tok).name in WRITER_SCRIPTS for tok in argv):
        return frozenset()
    explicit = _flag(argv, "--run-id")
    if explicit:
        return frozenset({explicit})
    cfg_tok = _flag(argv, "--config")
    cfg = _resolve(cfg_tok, cwd) if cfg_tok else None
    m = _RUN_ID_RE.search(_read_small(cfg)) if cfg is not None else None
    if not m:
        return frozenset()
    seed = _flag(argv, "--seed")
    return frozenset({seeded_run_id(m.group(1), seed) if seed is not None else m.group(1)})


def _open_paths(base: Path) -> tuple[str, ...]:
    try:
        fds = os.listdir(base / "fd")
    except OSError:
        return ()
    out = []
    for fd in fds:
        try:
            out.append(os.readlink(base / "fd" / fd))
        except OSError:
            continue
    return tuple(out)


def user_processes(proc_root: str = "/proc", exclude: Iterable[int] | None = None) -> list[Proc]:
    """Processes of the current user except `exclude` (default: the caller and its ancestors).
    Empty where there is no /proc (Windows): liveness cannot be checked there."""
    if not os.path.isdir(proc_root) or not hasattr(os, "getuid"):
        return []
    skip = ancestors(proc_root=proc_root) if exclude is None else set(exclude)
    uid = os.getuid()
    out = []
    for entry in os.listdir(proc_root):
        if not entry.isdigit() or int(entry) in skip:
            continue
        base = Path(proc_root, entry)
        try:
            if base.stat().st_uid != uid:
                continue
            raw = (base / "cmdline").read_bytes()
        except OSError:
            continue
        cmdline = raw.replace(b"\0", b" ").decode("utf-8", "replace").strip()
        if not cmdline:
            continue
        argv = [a.decode("utf-8", "replace") for a in raw.split(b"\0") if a]
        try:
            cwd = Path(os.readlink(base / "cwd"))
        except OSError:
            cwd = None
        out.append(Proc(int(entry), cmdline, expand_text(cmdline, cwd), writer_run_ids(argv, cwd),
                        _open_paths(base)))
    return out


def run_id_of(run_dir: str | os.PathLike[str]) -> str:
    return Path(os.path.abspath(run_dir)).name


def live_holders(run_dir: str | os.PathLike[str], procs: Sequence[Proc] | None = None) -> list[Proc]:
    """Processes writing this run: a live train.py / predict_oof.py for exactly this run id
    (`<run_id>_e8` is another run), or any process holding a file inside the run dir open.
    A process that only names the run (a later chain, a waiter, a scorer) does not hold it."""
    rid = run_id_of(run_dir)
    root = os.path.abspath(run_dir).rstrip(os.sep) + os.sep
    procs = user_processes() if procs is None else procs
    return [p for p in procs if rid in p.run_ids or any(o.startswith(root) for o in p.open_paths)]


def assert_not_live(run_dirs: Iterable[str | os.PathLike[str]], procs: Sequence[Proc] | None = None) -> None:
    procs = user_processes() if procs is None else procs
    errors = []
    for rd in run_dirs:
        for p in live_holders(rd, procs):
            errors.append(f"{rd}: live process {p.pid}: {p.cmdline[:160]}")
    if errors:
        raise RunGuardError("refusing to read a run that is still being written (architecture §4.1 "
                            "'never score a live chain'):\n  " + "\n  ".join(errors))


def is_full_run(run_dir: str | os.PathLike[str]) -> bool:
    """A full-data run (train.py --fold full) keeps its single member in <run>/full."""
    return (Path(run_dir) / FULL).is_dir()


def member_dirs(run_dir: str | os.PathLike[str], folds: Sequence[int]) -> list[Path]:
    """Member dirs to ship: [<run>/full] for a full-data run, else <run>/fold<k> for k in folds."""
    rd = Path(run_dir)
    return [rd / FULL] if is_full_run(rd) else [rd / f"fold{k}" for k in folds]


def smoke_reasons(run_dir: str | os.PathLike[str], folds: Sequence[int] = ()) -> list[str]:
    rid = run_id_of(run_dir)
    out = [f"run name {rid!r} matches {pat!r}" for pat in SMOKE_PATTERNS if fnmatch.fnmatch(rid, pat)]
    for md in member_dirs(run_dir, folds):
        m = _json_or_none(md / "metrics.json")
        if m and (m.get("limit") or m.get("synthetic")):
            out.append(f"{md.name} trained with limit={m.get('limit')} synthetic={m.get('synthetic')}")
    return out


def _json_or_none(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def fold_problems(fold_dir: str | os.PathLike[str]) -> list[str]:
    """Why this fold dir is not a finished training result ([] = finished)."""
    fd = Path(fold_dir)
    meta = _json_or_none(fd / "bundle.json")
    if meta is None:
        return [f"{fd}: no readable bundle.json (fold missing)"]
    probs = []
    if (fd / "last.pt").exists():
        probs.append(f"{fd}: last.pt present - training is running or was interrupted "
                     "(train_fold deletes it only after the final epoch)")
    cfg = meta.get("config") or {}
    # Bundles written before select_epoch existed (MVP era) always saved the best epoch.
    epochs, policy = cfg.get("epochs"), cfg.get("select_epoch") or "best"
    got = (meta.get("metrics") or {}).get("epoch")
    if epochs is None:
        return probs + [f"{fd}: bundle.json has no config.epochs"]
    final = int(epochs) - 1
    metrics = _json_or_none(fd / "metrics.json")
    last_seen = ((metrics or {}).get("last_epoch") or {}).get("epoch")
    if metrics is not None and last_seen is None and metrics.get("history"):
        last_seen = max(int(h["epoch"]) for h in metrics["history"])
    if policy == "last":
        if got != final:
            probs.append(f"{fd}: bundle epoch {got} != final epoch {final} (select_epoch=last)")
    else:
        best = ((metrics or {}).get("best_epoch") or {}).get("epoch", (metrics or {}).get("epoch"))
        if metrics is None:
            probs.append(f"{fd}: select_epoch={policy} needs metrics.json to prove completion")
        elif got != best:
            probs.append(f"{fd}: bundle epoch {got} != best epoch {best} (select_epoch={policy})")
    if metrics is not None and last_seen != final:
        probs.append(f"{fd}: metrics.json last epoch {last_seen} != final epoch {final}")
    return probs


def export_problems(run_dirs: Sequence[str | os.PathLike[str]], folds: Sequence[int], allow_smoke: bool = False,
                    procs: Sequence[Proc] | None = None) -> list[str]:
    """Everything that makes this set of runs unfit to ship; [] = OK to export.

    A missing fold is an error rather than a skip: members weigh 1.0 each, so an arm with 4 of
    5 folds would silently lose a fifth of its share of the ensemble."""
    procs = user_processes() if procs is None else procs
    errors = []
    for rd in run_dirs:
        if not Path(rd).is_dir():
            errors.append(f"{rd}: not a directory")
            continue
        errors += [f"{rd}: live process {p.pid}: {p.cmdline[:160]}" for p in live_holders(rd, procs)]
        if not allow_smoke:
            errors += [f"{rd}: smoke/sweep run ({r}); pass --allow-smoke to ship it anyway"
                       for r in smoke_reasons(rd, folds)]
        if is_full_run(rd):
            stray = sorted(p.name for p in Path(rd).glob("fold[0-9]*") if p.is_dir())
            if stray:
                errors.append(f"{rd}: holds both full/ and {stray}; a full-data run must hold only full/")
        for md in member_dirs(rd, folds):
            errors += fold_problems(md)
    return errors


def git_sha(root: str | os.PathLike[str]) -> str:
    """HEAD of the repo at root, suffixed "-dirty" when tracked files differ from it; "" when
    root is not a git checkout or git is missing."""
    try:
        sha = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True,
                             check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""
    return sha + ("-dirty" if dirty else "")
