"""Export / scoring guards (audit 2026-09-25 P0-2, P1-6): live runs, unfinished folds, missing
folds, smoke runs, and the git commit recorded in every exported bundle."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from rsna_kad import kaggle_export as ke
from rsna_kad import runguard
from rsna_kad.runguard import Proc
from test_infer import make_tiny_export

REPO = Path(__file__).resolve().parents[1]
HAS_PROC = os.path.isdir("/proc") and hasattr(os, "getuid")


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"script_{name}", REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_fold(run: Path, k: int, epochs: int = 10, epoch: int | None = None, last_pt: bool = False,
              policy: str = "last", best: int | None = None, limit=None) -> Path:
    fd = run / f"fold{k}"
    fd.mkdir(parents=True, exist_ok=True)
    epoch = epochs - 1 if epoch is None else epoch
    meta = {"run_id": run.name, "fold": k, "config": {"epochs": epochs, "select_epoch": policy},
            "metrics": {"epoch": epoch}, "git_sha": ""}
    (fd / "bundle.json").write_text(json.dumps(meta))
    last_seen = epoch if policy == "last" else epochs - 1
    (fd / "metrics.json").write_text(json.dumps({"epoch": epoch, "last_epoch": {"epoch": last_seen},
                                                 "best_epoch": {"epoch": epoch if best is None else best},
                                                 "limit": limit, "synthetic": False}))
    if last_pt:
        (fd / "last.pt").write_bytes(b"")
    return fd


# ------------------------------------------------------------------ fold completeness


def test_finished_fold_passes(tmp_path: Path):
    assert runguard.fold_problems(make_fold(tmp_path / "arm", 0)) == []


def test_fold_with_last_pt_is_refused(tmp_path: Path):
    probs = runguard.fold_problems(make_fold(tmp_path / "arm", 0, last_pt=True))
    assert len(probs) == 1 and "last.pt" in probs[0]


def test_fold_before_final_epoch_is_refused(tmp_path: Path):
    probs = runguard.fold_problems(make_fold(tmp_path / "arm", 0, epoch=4))
    assert any("bundle epoch 4 != final epoch 9" in p for p in probs)
    assert any("metrics.json last epoch 4" in p for p in probs)


def test_best_policy_needs_training_to_have_finished(tmp_path: Path):
    assert runguard.fold_problems(make_fold(tmp_path / "a", 0, epoch=3, policy="best")) == []
    fd = make_fold(tmp_path / "b", 0, epoch=3, policy="best")
    m = json.loads((fd / "metrics.json").read_text())
    m["last_epoch"]["epoch"] = 5
    (fd / "metrics.json").write_text(json.dumps(m))
    assert any("last epoch 5 != final epoch 9" in p for p in runguard.fold_problems(fd))
    fd = make_fold(tmp_path / "c", 0, epoch=3, policy="best", best=7)
    assert any("!= best epoch 7" in p for p in runguard.fold_problems(fd))


def test_legacy_bundle_without_select_epoch_is_judged_as_best(tmp_path: Path):
    """runs/mvp_effv2s_256 shape: no config.select_epoch, metrics.json without last/best_epoch."""
    def legacy(run: Path, history: int) -> Path:
        fd = run / "fold0"
        fd.mkdir(parents=True)
        (fd / "bundle.json").write_text(json.dumps({"config": {"epochs": 6}, "metrics": {"epoch": 2}}))
        (fd / "metrics.json").write_text(json.dumps({"epoch": 2, "history": [{"epoch": e} for e in range(history)]}))
        return fd
    assert runguard.fold_problems(legacy(tmp_path / "done", 6)) == []
    assert any("last epoch 3 != final epoch 5" in p for p in runguard.fold_problems(legacy(tmp_path / "cut", 4)))


def test_missing_fold_is_an_error_not_a_skip(tmp_path: Path):
    run = tmp_path / "main_arm"
    for k in (0, 1, 2, 4):
        make_fold(run, k)
    errors = runguard.export_problems([run], range(5), procs=[])
    assert len(errors) == 1 and "fold3" in errors[0] and "missing" in errors[0]


@pytest.mark.parametrize("name", ["armc_sweep_c1", "armd_sweep_d1g", "main_smoke_384", "armcp_smoke",
                                  "main_smoke_384_eager"])
def test_smoke_and_sweep_runs_need_allow_smoke(tmp_path: Path, name: str):
    run = tmp_path / name
    make_fold(run, 0)
    assert any("smoke/sweep" in e for e in runguard.export_problems([run], [0], procs=[]))
    assert runguard.export_problems([run], [0], allow_smoke=True, procs=[]) == []


def test_limit_trained_run_counts_as_smoke(tmp_path: Path):
    run = tmp_path / "main_arm"
    make_fold(run, 0, limit=200)
    assert any("limit=200" in e for e in runguard.export_problems([run], [0], procs=[]))


# ------------------------------------------------------------------ liveness


def _proc(pid: int, cmdline: str, cwd: Path, open_paths: tuple[str, ...] = ()) -> Proc:
    argv = cmdline.split()
    return Proc(pid, cmdline, runguard.expand_text(cmdline, cwd), runguard.writer_run_ids(argv, cwd), open_paths)


def test_live_holders_are_writers_of_the_run_not_mentions(tmp_path: Path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "eff_pseudo_cfg.yaml").write_text("run_id: eff_pseudo_256\nepochs: 8\n")
    (tmp_path / "configs" / "main_convnext_s_384_e8.yaml").write_text("run_id: main_convnext_s_384_e8\n")
    (tmp_path / "chain.sh").write_text('CFG=${CFG:-configs/eff_pseudo_cfg.yaml}\nRUN=main_armd_x\n')
    procs = [_proc(10, "python scripts/train.py --config configs/eff_pseudo_cfg.yaml --fold 0", tmp_path),
             _proc(11, "bash chain.sh", tmp_path),
             _proc(12, "python scripts/train.py --config configs/main_convnext_s_384_e8.yaml --fold 1", tmp_path),
             _proc(13, "python scripts/predict_oof.py --config configs/eff_pseudo_cfg.yaml", tmp_path),
             _proc(14, "python scripts/train.py --config configs/eff_pseudo_cfg.yaml --fold full --seed 2126",
                   tmp_path),
             _proc(15, "python scripts/train.py --config configs/eff_pseudo_cfg.yaml --fold 0 --run-id other_x",
                   tmp_path)]
    assert [p.pid for p in runguard.live_holders(tmp_path / "runs" / "eff_pseudo_256", procs)] == [10, 13]
    assert runguard.live_holders("runs/main_armd_x", procs) == []        # a chain that only names it
    # a longer run id that merely starts with this one is a different run
    assert runguard.live_holders("runs/main_convnext_s_384", procs) == []
    assert [p.pid for p in runguard.live_holders("runs/main_convnext_s_384_e8/", procs)] == [12]
    assert [p.pid for p in runguard.live_holders("runs/eff_pseudo_256_s2126", procs)] == [14]
    assert [p.pid for p in runguard.live_holders("runs/other_x", procs)] == [15]
    with pytest.raises(runguard.RunGuardError, match="live process 10"):
        runguard.assert_not_live([tmp_path / "runs" / "eff_pseudo_256"], procs)
    errors = runguard.export_problems([make_fold(tmp_path / "runs" / "eff_pseudo_256", 0).parent], [0],
                                      procs=procs)
    assert len(errors) == 2 and all("live process" in e for e in errors)


def test_open_file_inside_the_run_dir_holds_it(tmp_path: Path):
    run = tmp_path / "runs" / "arm_x"
    holder = _proc(20, "python some_tool.py", tmp_path, (str(run / "fold0" / "oof.parquet"),))
    other = _proc(21, "python some_tool.py", tmp_path, (str(tmp_path / "runs" / "arm_x_e8" / "a"),))
    assert [p.pid for p in runguard.live_holders(run, [holder, other])] == [20]


def test_false_positive_of_2026_09_27_a_later_chain_naming_a_finished_run(tmp_path: Path):
    """The refusals of C_p (hpe chain) and E_p2 (round-2 chain): each chain script named a
    finished run - in a comment, or as a config it would train later - while training another
    run. Only the train.py it runs holds anything."""
    (tmp_path / "qa" / "hpe").mkdir(parents=True)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "main_armc_p_hpe3.yaml").write_text("run_id: main_armc_p_hpe3\n")
    (tmp_path / "configs" / "main_armc_pseudo.yaml").write_text("run_id: main_armc_convnextv2_t_384_p\n")
    (tmp_path / "qa" / "hpe" / "chain.sh").write_text(
        "NEED_MIB=18000   # C_p peaks at 14.3 GB (runs/main_armc_convnextv2_t_384_p/fold*/train.log)\n"
        "c=configs/main_armc_p_hpe$n.yaml\npython scripts/train.py --config $c --fold $k\n"
        "# base recipe: configs/main_armc_pseudo.yaml\n")
    chain = _proc(30, "bash qa/hpe/chain.sh xfmr_q 3 5", tmp_path)
    trainer = _proc(31, "python scripts/train.py --config configs/main_armc_p_hpe3.yaml --fold 2", tmp_path)
    finished = tmp_path / "runs" / "main_armc_convnextv2_t_384_p"
    assert "main_armc_convnextv2_t_384_p" in chain.text          # what the old rule matched on
    assert runguard.live_holders(finished, [chain, trainer]) == []
    assert [p.pid for p in runguard.live_holders(tmp_path / "runs" / "main_armc_p_hpe3", [chain, trainer])] == [31]


def _wait_for(pred, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not pred() and time.monotonic() < deadline:
        time.sleep(0.1)


@pytest.mark.skipif(not HAS_PROC, reason="needs /proc")
def test_real_processes_writer_fd_holder_and_mention(tmp_path: Path):
    run = "guardprobe_run_%d" % os.getpid()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "train.py").write_text("import time; time.sleep(60)\n")
    (tmp_path / "cfg.yaml").write_text(f"run_id: {run}\n")
    rd = tmp_path / "runs" / run
    (rd / "fold0").mkdir(parents=True)
    (rd / "fold0" / "metrics.json").write_text("{}")
    sleep = "import time; time.sleep(60)"
    mention = subprocess.Popen([sys.executable, "-c", sleep, f"runs/{run}/fold0"], cwd=tmp_path)
    writer = subprocess.Popen([sys.executable, "scripts/train.py", "--config", "cfg.yaml", "--fold", "0"],
                              cwd=tmp_path)
    holder = subprocess.Popen([sys.executable, "-c", f"f = open({str(rd / 'fold0' / 'metrics.json')!r}); {sleep}"])
    try:
        want = sorted([writer.pid, holder.pid])
        _wait_for(lambda: sorted(p.pid for p in runguard.live_holders(rd)) == want)
        assert sorted(p.pid for p in runguard.live_holders(rd)) == want      # not the mere mention
    finally:
        for c in (mention, writer, holder):
            c.kill()
            c.wait()
    assert runguard.live_holders(rd) == []
    # A chain that scores its own run: the writer is the caller itself, which is excluded.
    (tmp_path / "scripts" / "train.py").write_text(
        "import sys; sys.path.insert(0, %r); from rsna_kad import runguard; "
        "print(len(runguard.live_holders(%r)))\n" % (str(REPO / "src"), str(rd)))
    out = subprocess.run([sys.executable, "scripts/train.py", "--config", "cfg.yaml"], cwd=tmp_path,
                         capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "0", out.stderr


# ------------------------------------------------------------------ scripts


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest() if path.exists() else ""


def test_export_script_refuses_before_writing_anything(tmp_path: Path, capsys: pytest.CaptureFixture):
    export_kaggle = _load_script("export_kaggle")
    notebook = REPO / "notebooks" / ke.NOTEBOOK_NAME
    before = _md5(notebook)
    run = tmp_path / "runs" / "main_arm"
    make_fold(run, 0)
    make_fold(run, 1, last_pt=True)
    rc = export_kaggle.main(["--run", str(run), "--folds", "0", "1", "2", "--name", "x", "--user", "u",
                             "--export-root", str(tmp_path / "export"), "--no-verify"])
    err = capsys.readouterr().err
    assert rc == 2 and "export refused" in err and "last.pt" in err and "fold2" in err
    assert not (tmp_path / "export").exists() and _md5(notebook) == before


def test_export_script_records_git_sha(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    make_tiny_export(tmp_path, n_folds=2)  # writes runs/tiny/fold{0,1} bundles
    run = tmp_path / "runs" / "tiny"
    for k in (0, 1):
        meta = json.loads((run / f"fold{k}" / "bundle.json").read_text())
        meta["metrics"]["epoch"] = meta["config"]["epochs"] - 1
        (run / f"fold{k}" / "bundle.json").write_text(json.dumps(meta))
    meta0 = json.loads((run / "fold0" / "bundle.json").read_text())
    meta0["git_sha"] = "trainsha"
    (run / "fold0" / "bundle.json").write_text(json.dumps(meta0))
    # The script regenerates the repo notebook and kernel metadata; the shipped copies stay untouched.
    monkeypatch.setattr(ke, "build_notebook", lambda p: Path(p))
    monkeypatch.setattr(ke, "write_kernel_metadata", lambda d, u: Path(d))
    export_kaggle = _load_script("export_kaggle")
    rc = export_kaggle.main(["--run", str(run), "--folds", "0", "1", "--name", "g", "--user", "u",
                             "--export-root", str(tmp_path / "export"), "--no-verify"])
    assert rc == 0
    sha = runguard.git_sha(REPO)
    doc = json.loads((tmp_path / "export" / "g" / "ensemble.json").read_text())
    assert doc["git_sha"] == sha
    f0 = json.loads((tmp_path / "export" / "g" / "models" / "tiny_f0.json").read_text())
    f1 = json.loads((tmp_path / "export" / "g" / "models" / "tiny_f1.json").read_text())
    assert f0["git_sha"] == "trainsha" and f1["git_sha"] == sha


@pytest.mark.skipif(subprocess.run(["git", "--version"], capture_output=True).returncode != 0,
                    reason="git missing")
def test_git_sha_clean_dirty_and_outside_a_repo(tmp_path: Path):
    repo = tmp_path / "r"
    repo.mkdir()
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.invalid"]
    subprocess.run(git + ["init", "-q"], check=True)
    (repo / "f.txt").write_text("a\n")
    subprocess.run(git + ["add", "f.txt"], check=True)
    subprocess.run(git + ["commit", "-q", "-m", "x"], check=True)
    head = subprocess.run(git + ["rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    assert runguard.git_sha(repo) == head
    (repo / "untracked.txt").write_text("u\n")
    assert runguard.git_sha(repo) == head
    (repo / "f.txt").write_text("b\n")
    assert runguard.git_sha(repo) == head + "-dirty"
    assert runguard.git_sha(tmp_path / "nothing") == ""
