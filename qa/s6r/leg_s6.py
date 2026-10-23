#!/usr/bin/env python3
# leg_s6.py: the S6-only inference leg, derived from nartaa's public rsna-knee-s6fall-solo.py (Apache-2.0). The original
# header and docstring below describe the A+B K1 kernel and are kept verbatim; what this leg changes:
# - component A (the 0.949 CoAtNet) is gone: no embedded source, no A stage, no rank-correlation sanity floor against it.
# - a study S6 still fails on after the retry is filled with the column mean of the scored studies before ranking (the
#   original's B-side fill in blend_with_failures; its rank(A) override has nothing to take from). Up to
#   max(B_FAIL_MIN, ceil(B_FAIL_FRAC * N)) such studies at any N, more is a hard error (the model, not a study, is broken).
# - the checkpoint (S6_MODEL_FILE / S6_MODEL_SHA256 / S6_ARM ...) and DS_SLUG are plain constants; the DONE marker is optional.
# - GUARD_MAX_S is this leg's own hang guard (default 4 h); B gets everything it leaves minus END_SLACK_S.
# - a parent CUDA_VISIBLE_DEVICES is honoured: worker gi runs on its gi-th entry, not on physical GPU gi.
# RSNA Knee: S6 ConvNeXt (all data), 0.950 public. One click: Copy & Edit, then Submit. Needs a T4 GPU, internet off.
# Inputs (all public): nartaa/rsna-knee-s6fall-weights (our weights + code manifest), heliosli/rsna-orthofoundation-fold0-
# epoch7-20261004 (Heliosli's SKM-TEA anatomy V-Net + data pipeline), nartaa/rsna-knee-publication-swa-weights-20261007 (our
# 0.949 CoAtNet, used only for a study the ConvNeXt cannot read). Version 1 scored 0.950 reading the same files from a private
# bundle; from version 2 every file is read from these public datasets and checked against the scored sha256 before use.
# Write-up: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/749359
"""K1: per-finding percentile-rank blend, final = 0.7 x rank(A) + 0.3 x rank(B), ties averaged (= gate_score.rank_blend).

A = the scored 0.949 single model (submission 56906959): its kernel source, sha256 bcd864cb...79c5, is embedded below byte for
    byte and run UNCHANGED as its own process.  Its /kaggle/working/submission.csv is moved into k1/A/ the moment the process
    exits, so a single-model file can never be left where Kaggle would score it.
B = S6F0 (convnext_small, v4 views placed by the SKM-TEA anatomy V-Net, blendjev labels): Heliosli's compact_infer.py (sha256
    34f433fe..., unchanged) through a launcher equal to S6F0/infer_s6f0.py, with the laptop OOF arguments; one process per GPU.
    The one deliberate argument difference: --amp fp16 (laptop bf16), because the T4 (sm_75) has no bf16 (see PREREG_K1.md).
Every component failure, missing study, non-finite value or receipt mismatch is a HARD ERROR: submission.csv is deleted, a
reason is written to k1/k1_error.json and the kernel raises.  There is no fallback to one component.  Two capped per-study
allowances, both the deployed kernels' own behaviour (PREREG_K1.md section 5): A's internal fallbacks (<= max(3, 1 % of N)) and,
on the hidden test only (N >= 100), B studies that still fail after the retry (<= max(3, 0.25 % of N)); such a row takes
rank(A) (the 0.954 notebook's rule).  Above either cap: hard error.
Budgets: A gets A_TIMEOUT_S per attempt, with one retry after a stall (log silent for A_STALL_S) or a timeout when guard time
allows; A can never eat into B's floor B_TIMEOUT_S; B gets everything the guard has left (minus END_SLACK_S).
Hang guard: a watchdog thread kills every child process group, deletes submission.csv and _exit(1)s at GUARD_S; a C-level
faulthandler.dump_traceback_later backstop fires BACKSTOP_EXTRA_S later in case the interpreter is wedged.
MODE "parity" (a separate private kernel) first runs B on PARITY_IDS (60 seeded fold-0 TRAIN studies, every series intact) with a
pixel-hash dump, then the normal path on the public test studies (B there with the same dump).  Built from k1_kernel_template.py
by build_k1.py, which bakes K1_BUILD_ID: every receipt (k1_receipt.json / k1_error.json) echoes it with this execution's own
identity, so the downloaded evidence can be bound to the pushed source (PREREG_K1.md section 4, provenance)."""
MODE = "submit"   # "submit" (nartaa/rsna-knee-k1-0949-s6f0-blend) or "parity" (nartaa/rsna-knee-k1-s6f0-parity)
K1_BUILD_ID = "2026-10-12T10:12:34+03:00/f0d87208a3de20a5ac936388bfd763799d6391833fea3a4f90ec6de14ef5051b"   # "<build time>/<sha256(build time + newline + this file with this value as the placeholder)>"

import faulthandler, hashlib, json, math, os, re, shutil, signal, subprocess, sys, tarfile, threading, time, traceback
from pathlib import Path
import numpy as np
import pandas as pd

KROOT = Path(os.environ.get("K1_KAGGLE_ROOT", "/kaggle"))      # local-test hook only; never set on Kaggle
INPUT, WORK = KROOT / "input", KROOT / "working"
OUT = WORK / "leg_s6"
TMP = Path(os.environ.get("K1_TMP", "/tmp/leg_s6"))
COMP = "rsna-knee-abnormality-detection"
DS_SLUG = "rsna-knee-s6fall-weights"   # ours later: "rsna-kad-s6r" (owner sroziews); find_mount resolves either input layout
HELIOS_SLUG = "rsna-orthofoundation-fold0-epoch7-20261004"   # public: Heliosli's anatomy.pt + cloud_code (attached, not copied)
EXTRAS_FILE = "s6_extras.bin"
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
       "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

# ---- the S6 checkpoint: swap these to run another model through the same pipeline ------------------------------------------
# S6_ARM is compact_infer's --arm and must equal run_config["arms"]. S6_DONE_FILE None: no DONE marker in the dataset, the
# epochs / train-studies checks are then skipped.
S6_MODEL_FILE = "E_prior_s42_fall.pt"
S6_MODEL_SHA256 = "0c7297d58f061ab4e5d7c55046f51fa3e8934928046f0647a0c7854e98371c12"
S6_ARM = "E_prior"
S6_RUN_CONFIG = "E_prior_s42_fall.run_config.json"
S6_DONE_FILE = "E_prior_s42_fall.DONE.json"
S6_EXPECT_EPOCHS, S6_EXPECT_TRAIN = 16, 4263
B_MODEL = {"file": S6_MODEL_FILE, "source_file": S6_MODEL_FILE, "sha256": S6_MODEL_SHA256, "run_config": S6_RUN_CONFIG,
           "done": S6_DONE_FILE, "arm": S6_ARM, "expect_epochs": S6_EXPECT_EPOCHS, "expect_train": S6_EXPECT_TRAIN}
ANATOMY_FILE, ANATOMY_SHA256 = "anatomy.pt", "5904dd16eb9a4fd7a1caf90ac977f8fe2ef364908598c23c209db3fe7fd1563e"
PAYLOAD_FILE, PAYLOAD_MAGIC = "k1_code_payload.bin", b"K1PAYLOAD1\n"
SOURCE_SHA = {                                                 # the 0.954 notebook's SOURCE_SHA = S6F0/s6common.SOURCE_SHA
    "px_cache.py": "b13e38a7ad3bd496f8027e02209653e224b0926db4f0853eeb0d076a2d6e2632",
    "compact_train.py": "14767c7a8f0daa986477511fe9a83aefe032a208ae978fabf15a5f896992ac80",
    "compact_infer.py": "34f433fe72b732543813165c3397ec80cef01b0d49b25b7b22668714f0b0ef4a",
    "probe_infer.py": "a6cc31448019c84be5868d0a2ff1d99719b23b0f86110efc03efff3ce6bd6980",
    "knee_anchor_bank_v2.py": "e564807a129dfcd3c984348917436ee6083066a4d685340587ebd1d3ea14f309",
    "side_common.py": "bbcc5d027834e6a6fd48c4a8cf228e25661b7522c792587c840fe4d7cd5c0566",
}
B_RECEIPT_SOURCES = ("px_cache.py", "compact_train.py", "compact_infer.py", "probe_infer.py", "knee_anchor_bank_v2.py")
                                 # = compact_infer's own receipt sha list for --views v4 without a side model: all must be present
CODE_SHA = {                                                   # the 0.954 notebook's CODE_SHA (cloud_code/training pins)
    "native_cache.py": "c53ce8985406b3b918d9e821d43518479e89d9f946daebd817f378e3949718b9",
    "native_data.py": "f59afe205c8e193369fa3ccd22edcce8df3b293205fc48ddb1dd45169ee43b36",
    "input_provenance.py": "b3a22500c7f976f6ee040239ada5cacecf2326861cb49af1c7d8cfdeeb55fcda",
    "intensity_mapping.py": "19be2d8cbd6f8971c1288d7512a8cca41d143470c2ba6e9b9096fde035fa9d41",
    "coronal_views.py": "841f216d2682ae2c0f8afd82fc902b9fa74924d7145b952069dea8b0bbd03c40",
    "exact_duplicates.py": "51b38db21ce9c5fe2b6278a835983414756b61ac6af81aba380b44b41b075897",
    "dicom_geometry.py": "12aa501f9704aef15b7a3bbd560951acb6d95b89b7de0141d6cf89fc3f051ab1",
    "common.py": "8283fa1f8e7ef8cfacbba52df1715636be3c9ad688e39912814e7d5bcf57105f",
}
B_GEOMETRY = {"backbone": "convnext_small", "size": 320, "crop": 288, "views": "v4", "slices": 3, "phys_mm": 3.5,
              "eval_stride": 1, "chunk": 192, "item_timeout": 900}             # = S6F0/oof_step.py, checked vs run_config.json
B_AMP = "fp16"
B_WORKERS = 2                    # DataLoader workers per GPU process (Heliosli's 0.954 notebook: 2; laptop OOF: 3, speed only)
B_FAIL_MIN, B_FAIL_FRAC = 3, 0.02     # studies still failing after the retry get the column-mean fill; above
                                      # max(3, ceil(2 % of N)) (26 at N = 1,284) the leg fails hard instead

# ---- budgets: this leg runs beside others inside Kaggle's 9 h, so its guard is its own share, not the whole notebook --------
GUARD_MAX_S = 4 * 3600           # this leg's hang guard; the faulthandler backstop fires BACKSTOP_EXTRA_S after it
SETUP_SLACK_S = 30 * 60          # env probe, inputs, B probe, write-out, END_SLACK_S
B_TIMEOUT_S = GUARD_MAX_S - SETUP_SLACK_S    # B's floor (a warning only when less is left); B gets all the guard leaves
PARITY_TIMEOUT_S = 3600
END_SLACK_S = 10 * 60            # B's budget = guard time left - this (blend + receipts before the guard)
EXIT_GRACE_S = 120               # outputs complete (A: DONE + submission.csv; B: WORKER_DONE + receipt + logits) but no exit yet
B_DEADLINE_MARGIN_S = 20 * 60    # compact_infer --deadline = B start + B budget - this (> --item-timeout 900 s: a worker hang at
                                 # the deadline ends in compact_infer, before our kill)
B_RETRY_MIN_S = 10 * 60          # a fallback retry runs only if at least this much of the B budget is left
GUARD_S = GUARD_MAX_S + (PARITY_TIMEOUT_S if MODE == "parity" else 0)
assert B_TIMEOUT_S + SETUP_SLACK_S <= GUARD_MAX_S and B_TIMEOUT_S > B_DEADLINE_MARGIN_S, "budgets exceed the hang guard"
BACKSTOP_EXTRA_S = 10 * 60
HEARTBEAT_S = 300
PARITY_IDS = []

T0 = time.time()
_LOG = None
_GUARD_FILE = None
_CHILDREN = {}                   # pgid -> name
_GUARD_STOP = threading.Event()  # set on success: a lingering interpreter (notebook-style runner) must not fire the guard later
# this execution: a fresh nonce + the start time (UTC); the controller accepts the evidence only from a run that started after
# its push intent and before its fetch, of the accepted version (review C4).  KAGGLE_KERNEL_RUN_TYPE only, never other KAGGLE_*
_RUN = {"nonce": os.urandom(16).hex(), "started_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(T0)),
        "kaggle_run_type": os.environ.get("KAGGLE_KERNEL_RUN_TYPE")}
_RECEIPT = {"mode": MODE, "build_id": K1_BUILD_ID, "run": _RUN, "stages": {}, "gpu": {}}


class K1Error(RuntimeError):
    pass


class K1Timeout(K1Error):
    """a stage hit its time cap, or (A attempt 1) its log stalled"""


def log(*a):
    msg = "%s +%6.0fs %s" % (time.strftime("%H:%M:%S", time.gmtime()), time.time() - T0, " ".join(str(x) for x in a))
    print(msg, flush=True)
    if _LOG is not None:
        try:
            _LOG.write(msg + "\n"); _LOG.flush()
        except Exception:
            pass


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def build_id_of(text):
    """K1_BUILD_ID recomputed from a run.py text (= build_k1.build_id_of = compare_parity.build_id_of): '<at>/<sha256(at + newline
    + the text with the constant's value put back to the template placeholder)>'; None if the text has no built id line."""
    m = re.search(r'^K1_BUILD_ID = "([^"/]+)/([0-9a-f]{64})"', text, re.M)
    if m is None:
        return None
    base = text[:m.start()] + 'K1_BUILD_ID = "__K1_' + 'BUILD_ID__"' + text[m.end():]     # split: the template holds one placeholder
    return m.group(1) + "/" + hashlib.sha256((m.group(1) + "\n" + base).encode()).hexdigest()


def self_identity():
    """This execution's own source, when the runner exposes it (Kaggle runs a script as /kaggle/src/script.py; a notebook-style
    wrapper may define no __file__): its path, sha256 and the build id recomputed from its text (derived_matches: that id equals
    the baked K1_BUILD_ID).  Only a file holding this build's K1_BUILD_ID line counts.  Never raises; None when unavailable."""
    out = {"path": None, "sha256": None, "derived_build_id": None, "derived_matches": None}
    for cand in (globals().get("__file__"), sys.argv[0] if sys.argv else None):
        try:
            if not cand or not os.path.isfile(cand):
                continue
            text = Path(cand).read_text(errors="replace")
            if ('K1_BUILD_ID = "%s"' % K1_BUILD_ID) not in text:
                continue
            out.update(path=str(cand), sha256=sha256_file(cand), derived_build_id=build_id_of(text))
            out["derived_matches"] = out["derived_build_id"] == K1_BUILD_ID
            break
        except Exception as e:
            out["error"] = repr(e)[:200]
    return out


def save_json(path, obj):
    path = Path(path); tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str)); os.replace(tmp, path)


def tail(path, n=3000):
    try:
        return Path(path).read_text(errors="replace")[-n:]
    except Exception as e:
        return "(no log: %r)" % e


def remove_submission():
    for p in (WORK / "submission.csv", WORK / ".submission.csv.tmp"):
        try:
            p.unlink()
        except FileNotFoundError:
            pass


def gpu_snapshot():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=20)
        return [l.strip() for l in r.stdout.splitlines() if l.strip()] or [r.stderr.strip()[:200]]
    except Exception as e:
        return ["nvidia-smi failed: %r" % e]


def mem_available_mb():
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    except Exception:
        pass
    return -1


def group_rss_mb(pgid):
    tot = 0
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            st = open("/proc/%s/stat" % d).read()
            f = st[st.rindex(")") + 2:].split()
            if int(f[2]) == pgid:
                tot += int(f[21]) * os.sysconf("SC_PAGE_SIZE")
        except Exception:
            continue
    return tot // (1 << 20)


def kill_group(pgid):
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    _CHILDREN.pop(pgid, None)


def kill_all_children():
    for pgid in list(_CHILDREN):
        kill_group(pgid)


def stage(name):
    _RECEIPT["stages"][name] = {"start_s": round(time.time() - T0, 1)}
    _RECEIPT["gpu"][name + ".start"] = gpu_snapshot()
    log("[stage] %s start | gpu %s | MemAvailable %d MB" % (name, _RECEIPT["gpu"][name + ".start"], mem_available_mb()))


def stage_done(name, **info):
    s = _RECEIPT["stages"][name]
    s["seconds"] = round(time.time() - T0 - s["start_s"], 1)
    s.update(info)
    _RECEIPT["gpu"][name + ".end"] = gpu_snapshot()
    log("[stage] %s done in %.0fs | gpu %s" % (name, s["seconds"], _RECEIPT["gpu"][name + ".end"]))


# ---- hang guard --------------------------------------------------------------------------------------------------------------
def _watchdog(cap_s):
    deadline = T0 + cap_s
    while True:
        left = deadline - time.time()
        if left <= 0:
            break
        if _GUARD_STOP.wait(min(left, 15)):
            return
    try:
        n_children = len(_CHILDREN)
        kill_all_children()                                  # cleanup first: a failing dump below must not skip it
        remove_submission()
        log("[guard] FIRED after %.0fs (cap %ds): killed %d child group(s), deleted submission.csv; dumping every thread"
            % (time.time() - T0, cap_s, n_children))
        for target in (_GUARD_FILE, sys.stdout):
            try:
                faulthandler.dump_traceback(file=target, all_threads=True)
            except Exception as e:
                log("[guard] dump to %r failed: %r" % (target, e))
        try:
            save_json(OUT / "k1_error.json", {"error": "hang guard fired", "cap_s": cap_s, "elapsed_s": round(time.time() - T0, 1),
                                              "receipt": _RECEIPT})
        except Exception:
            pass
        sys.stdout.flush()
    finally:
        os._exit(1)


def arm_guard(cap_s):
    global _GUARD_FILE
    _GUARD_FILE = open(OUT / "k1_hang_guard_stacks.txt", "w")
    faulthandler.enable(file=_GUARD_FILE, all_threads=True)
    faulthandler.dump_traceback_later(cap_s + BACKSTOP_EXTRA_S, repeat=False, file=_GUARD_FILE, exit=True)
    threading.Thread(target=_watchdog, args=(cap_s,), name="k1-guard", daemon=True).start()
    log("[guard] armed: watchdog %ds (kill children, delete submission.csv, _exit 1); faulthandler backstop %ds"
        % (cap_s, cap_s + BACKSTOP_EXTRA_S))


# ---- bounded subprocess groups -----------------------------------------------------------------------------------------------
def guard_left():
    return T0 + GUARD_S - time.time()


def _size(path):
    try:
        return os.path.getsize(path)
    except OSError:
        return -1


def run_group(specs, timeout_s, what):
    """Start every spec {name, cmd, env, cwd, log[, done_when, stall_s]} in its own session; poll until all exit, one fails, or
    timeout_s passes.  A failure, a stall or the timeout kills every group (DataLoader / Pool children included).
    done_when(): the child's outputs are complete; if it then has not exited within EXIT_GRACE_S (an interpreter-exit hang), its
    group is killed and the run counts as a success.  stall_s: the log has not grown for that long = K1Timeout.
    Returns [(name, rc, seconds, note)]."""
    procs, t_start, last_hb, last_check = [], time.time(), time.time(), 0.0
    try:
        for s in specs:
            fh = open(s["log"], "w")
            p = subprocess.Popen(s["cmd"], env=s["env"], cwd=s.get("cwd"), stdout=fh, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
            _CHILDREN[p.pid] = s["name"]
            procs.append({"s": s, "p": p, "fh": fh, "t1": time.time(), "size": -1, "changed": time.time(), "done_at": None})
            log("[%s] started %s pid %d" % (what, s["name"], p.pid))
        done = {}
        while len(done) < len(procs):
            now_ = time.time()
            check = now_ - last_check >= 5
            if check:
                last_check = now_
            for r in procs:
                s, p = r["s"], r["p"]
                if s["name"] in done:
                    continue
                rc, note = p.poll(), ""
                if rc is None and check and s.get("done_when") is not None:
                    try:
                        complete = bool(s["done_when"]())
                    except Exception:
                        complete = False
                    if complete and r["done_at"] is None:
                        r["done_at"] = now_
                    if r["done_at"] is not None and now_ - r["done_at"] > EXIT_GRACE_S and p.poll() is None:
                        kill_group(p.pid)
                        rc, note = 0, "outputs complete, no exit after %ds: group killed, accepted" % EXIT_GRACE_S
                        log("[%s] %s: %s" % (what, s["name"], note))
                if rc is not None:
                    done[s["name"]] = (rc, round(time.time() - r["t1"], 1), note)
                    r["fh"].close(); kill_group(p.pid)
                    log("[%s] %s exited rc %d after %.0fs" % (what, s["name"], rc, time.time() - r["t1"]))
                    if rc != 0:
                        for r2 in procs:
                            if r2["s"]["name"] not in done:
                                kill_group(r2["p"].pid); log("[%s] killed %s (a sibling failed)" % (what, r2["s"]["name"]))
                        raise K1Error("%s: %s exited %d; log tail:\n%s" % (what, s["name"], rc, tail(s["log"])))
                    continue
                if check and s.get("stall_s"):
                    sz = _size(s["log"])
                    if sz != r["size"]:
                        r["size"], r["changed"] = sz, now_
                    elif now_ - r["changed"] > s["stall_s"]:
                        for r2 in procs:
                            if r2["s"]["name"] not in done:
                                kill_group(r2["p"].pid)
                        raise K1Timeout("%s: %s stalled: log unchanged for %ds (at %d B); log tail:\n%s" % (
                            what, s["name"], s["stall_s"], sz, tail(s["log"], 1500)))
            if len(done) == len(procs):
                break
            if time.time() - t_start > timeout_s:
                for r in procs:
                    if r["s"]["name"] not in done:
                        kill_group(r["p"].pid)
                raise K1Timeout("%s: timeout after %ds; still running %s" % (
                    what, timeout_s, [r["s"]["name"] for r in procs if r["s"]["name"] not in done]))
            if time.time() - last_hb >= HEARTBEAT_S:
                last_hb = time.time()
                for r in procs:
                    if r["s"]["name"] in done:
                        continue
                    prog = [l for l in tail(r["s"]["log"], 4000).splitlines() if l.startswith("{") or "[gpu]" in l or "[pre]" in l][-1:]
                    log("[heartbeat] %s %s | %.0fs | rss %d MB | MemAvailable %d MB | gpu %s | %s" % (
                        what, r["s"]["name"], time.time() - r["t1"], group_rss_mb(r["p"].pid), mem_available_mb(), gpu_snapshot(),
                        prog[0][:300] if prog else "(no progress line yet)"))
            time.sleep(2)
        return [(r["s"]["name"],) + done[r["s"]["name"]] for r in procs]
    finally:
        for r in procs:
            if r["p"].poll() is None:
                kill_group(r["p"].pid)
            try:
                r["fh"].close()
            except Exception:
                pass


# ---- environment and inputs --------------------------------------------------------------------------------------------------
PROBE = r'''
import json, sys, importlib.util
out = {"python": sys.version.split()[0]}
import torch
out.update(torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
           cuda_available=torch.cuda.is_available(), n_gpu=torch.cuda.device_count())
if torch.cuda.is_available():
    out["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    out["capability"] = list(torch.cuda.get_device_capability(0))
    out["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
for m in ("numpy", "pandas", "timm", "pydicom", "cv2", "scipy", "safetensors"):
    try:
        out[m] = getattr(__import__(m), "__version__", "?")
    except Exception as e:
        out[m] = "MISSING " + type(e).__name__
for m in ("requests", "kagglesdk", "pylibjpeg", "gdcm"):
    out["has_" + m] = importlib.util.find_spec(m) is not None
print("K1_ENV " + json.dumps(out), flush=True)
'''


def probe_env():
    r = subprocess.run([sys.executable, "-c", PROBE], capture_output=True, text=True, timeout=300, stdin=subprocess.DEVNULL)
    line = [l for l in r.stdout.splitlines() if l.startswith("K1_ENV ")]
    if r.returncode != 0 or not line:
        raise K1Error("environment probe failed rc %d: %s %s" % (r.returncode, r.stdout[-1500:], r.stderr[-1500:]))
    return json.loads(line[0][len("K1_ENV "):])


def find_mount(slug, marker):
    cands = [INPUT / slug, INPUT / "competitions" / slug]
    cands += list(INPUT.glob("datasets/*/" + slug)) + list(INPUT.glob("datasets/*/" + slug + "/*"))
    cands += list(INPUT.glob("*/" + slug)) + list(INPUT.glob(slug + "/*"))
    found = []
    for p in cands:
        if (p / marker).is_file() and p.resolve() not in [f.resolve() for f in found]:
            found.append(p)
    if len(found) != 1:
        raise K1Error("cannot resolve %s (marker %s): %s" % (slug, marker, found))
    return found[0]


def safe_rel(name):
    p = Path(name)
    if p.is_absolute() or ".." in p.parts or not p.parts:
        raise K1Error("unsafe payload member %r" % name)
    return p


def extract_payload(path, target, members):
    """The code payload = PAYLOAD_MAGIC + an uncompressed tar (opaque to Kaggle's archive handling); exact member set, sizes, sha."""
    if target.exists():
        shutil.rmtree(target)
    with open(path, "rb") as f:
        if f.read(len(PAYLOAD_MAGIC)) != PAYLOAD_MAGIC:
            raise K1Error("payload magic mismatch")
        with tarfile.open(fileobj=f, mode="r:") as tar:
            ms = tar.getmembers()
            if {m.name for m in ms} != set(members) or len(ms) != len(members):
                raise K1Error("payload member set differs from MANIFEST")
            for m in ms:
                safe_rel(m.name)
                if not m.isfile() or m.size != members[m.name]["bytes"]:
                    raise K1Error("payload member %s: not a regular file or size differs" % m.name)
            for m in ms:
                dst = target / safe_rel(m.name)
                dst.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(m) as src, open(dst, "xb") as out:
                    shutil.copyfileobj(src, out, 1 << 20)
    for name, rec in members.items():
        if sha256_file(target / name) != rec["sha256"]:
            raise K1Error("payload member %s sha differs" % name)


LAUNCHER = r'''"""K1 component-B launcher = S6F0/infer_s6f0.py with Kaggle paths.  Runs Heliosli's compact_infer.main() UNCHANGED.
As on the laptop: sys.path = [cloud_code, heliosli, ...] and the training.download_data stub (only the chunked-archive reader
uses it; we read the DICOM directory), installed in the main process AND in every spawn worker (CStudies._setup).
Three K1 guards, none of which changes a prediction on the laptop's path (each acts only where that path would fail):
- distutils: the vendored MONAI 0.6.0 imports distutils.util.strtobool (Python >= 3.12 has no stdlib distutils; setuptools'
  shim provides it on the laptop).  Only if that import fails, a module holding the stdlib's strtobool is installed.
- setup failure (review M1): compact_infer's _setup sets self.native before it loads the SKM-TEA V-Net, so a failed load left a
  worker scoring every later study with field-centre views and no anchor, silently.  Here a failed _setup resets self.native
  (the next study re-runs it; probe_infer.install_native is memoised, the one step that is not re-entrant) and re-raises (this
  study becomes a fallback, retried); a scored item without an anchor is turned into a failed item (fallback, retried).
K1_PX_DUMP (parity mode only): every item's pixel / meta hashes are appended to <dir>/px_<pid>.jsonl; the item is not changed."""
import hashlib, json, os, sys, types
SRC = __SRC__
CODE = __CODE__
ANCHOR_VIEWS = ("v3k", "v4")


def _strtobool(val):
    """distutils.util.strtobool (CPython 3.11 / setuptools._distutils), unchanged semantics."""
    val = val.lower()
    if val in ("y", "yes", "t", "true", "on", "1"):
        return True
    elif val in ("n", "no", "f", "false", "off", "0"):
        return False
    else:
        raise ValueError(f"invalid truth value {val!r}")


def install_distutils():
    if "distutils.util" in sys.modules:
        return
    try:
        import distutils.util  # noqa: F401  (setuptools' shim on Python >= 3.12)
    except Exception:
        d, u = types.ModuleType("distutils"), types.ModuleType("distutils.util")
        d.__path__ = []
        u.strtobool = _strtobool
        d.util = u
        sys.modules["distutils"], sys.modules["distutils.util"] = d, u
        os.environ["K1_DISTUTILS_SHIM"] = "1"


def install_paths():
    sys.dont_write_bytecode = True
    for p in (SRC, CODE):
        if p not in sys.path:
            sys.path.insert(0, p)
    if "training.download_data" not in sys.modules:
        m = types.ModuleType("training.download_data")
        m.ChunkedArchive = None
        sys.modules["training.download_data"] = m
    install_distutils()


def memoise_install_native():
    import probe_infer as PI
    if getattr(PI, "_k1_memo", False):
        return
    orig, cache = PI.install_native, {}

    def install_native(lean=True):
        if lean not in cache:
            cache[lean] = orig(lean)
        return cache[lean]
    PI.install_native = install_native
    PI._k1_memo = True


install_paths()
import compact_infer as CI

assert os.path.dirname(os.path.abspath(CI.__file__)) == SRC, CI.__file__
PX_DUMP = os.environ.get("K1_PX_DUMP")


class LocalCStudies(CI.CStudies):
    def _setup(self):
        install_paths()
        memoise_install_native()
        try:
            super()._setup()
            if self.views in ANCHOR_VIEWS and self.bank is None:
                raise RuntimeError("K1: the SKM-TEA anchor bank did not load")
        except BaseException:
            self.native = None
            raise

    def __getitem__(self, i):
        out = super().__getitem__(i)
        if out.get("error") is None and self.views in ANCHOR_VIEWS and out.get("anchor") is None:
            out = dict(out, error="K1: item built without an SKM-TEA anchor", images=out["images"][:0], meta=out["meta"][:0])
        if PX_DUMP:
            try:
                img = out["images"].numpy()
                meta = out["meta"].numpy()
                rec = {"uid": out["uid"], "error": out["error"], "n": int(len(img)),
                       "img_sha256": hashlib.sha256(img.tobytes()).hexdigest(),
                       "meta_sha256": hashlib.sha256(meta.astype("float32").tobytes()).hexdigest(),
                       "win_sha1_10": [hashlib.sha1(w.tobytes()).hexdigest()[:10] for w in img],
                       "img_mean": float(img.mean()) if len(img) else None, "anchor": out.get("anchor"),
                       "seconds": out.get("seconds")}
            except Exception as exc:
                rec = {"uid": out.get("uid"), "hash_error": repr(exc)[:200]}
            with open(os.path.join(PX_DUMP, "px_%d.jsonl" % os.getpid()), "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
        return out


CI.CStudies = LocalCStudies

if __name__ == "__main__":
    CI.main()
'''


B_PROBE = r'''
import importlib.util, json, os, sys, time
launcher, anatomy, model, backbone, crop, slices = sys.argv[1:7]
t0 = time.time()
out = {}
try:
    import setuptools
    out["setuptools"] = setuptools.__version__
except Exception as e:
    out["setuptools"] = "MISSING " + type(e).__name__
out["has_distutils_hack"] = importlib.util.find_spec("_distutils_hack") is not None
spec = importlib.util.spec_from_file_location("k1_b_launcher_probe", launcher)
L = importlib.util.module_from_spec(spec)
spec.loader.exec_module(L)                      # the launcher's own path setup, stubs and distutils guard; imports compact_infer
import distutils.util, torch
out["distutils"] = "K1 shim" if os.environ.get("K1_DISTUTILS_SHIM") else str(getattr(sys.modules["distutils"], "__file__", "?"))
import knee_anchor_bank_v2 as KB
seg = KB.load_model(anatomy, "cpu")              # the vendored MONAI 0.6.0 V-Net + anatomy.pt, as every B worker loads it
import monai
out["monai"] = [monai.__version__, os.path.relpath(monai.__file__, L.CODE)]
with torch.no_grad():
    y = seg(torch.zeros(1, 1, 128, 128))
out["vnet_out"] = list(y.shape)
import timm, compact_train as CT
out["timm"] = timm.__version__
sd = torch.load(model, map_location="cpu", weights_only=True)
spatial = any(k == "sp_q" or k.endswith(".sp_q") for k in sd)
net = CT.Net(backbone, None, freeze_blocks=16, crop=int(crop), state_spec=None, in_chans=int(slices), spatial=spatial)
net.load_state_dict(sd, strict=True)            # = compact_infer.load_group for this model
out.update(spatial=spatial, params=int(sum(p.numel() for p in net.parameters())), seconds=round(time.time() - t0, 1))
print("K1_BPROBE " + json.dumps(out), flush=True)
'''


def probe_b_stack(I):
    """B's own stack before anything is spent on A: the launcher (stubs, distutils guard), compact_infer import, the vendored
    MONAI V-Net with anatomy.pt (+ one forward pass), compact_train.Net with the model's state dict (strict).  CPU only."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1")
    env.pop("K1_DISTUTILS_SHIM", None)
    g = B_GEOMETRY
    r = subprocess.run([sys.executable, "-c", B_PROBE, str(I.launcher), str(I.anatomy), str(I.models / B_MODEL["file"]),
                        g["backbone"], str(g["crop"]), str(g["slices"])],
                       capture_output=True, text=True, timeout=900, stdin=subprocess.DEVNULL, env=env, cwd=str(TMP))
    line = [l for l in r.stdout.splitlines() if l.startswith("K1_BPROBE ")]
    if r.returncode != 0 or not line:
        raise K1Error("component-B stack probe failed rc %d: %s %s" % (r.returncode, r.stdout[-1500:], r.stderr[-2500:]))
    return json.loads(line[0][len("K1_BPROBE "):])


def write_launcher(src_dir, code_dir, path):
    text = LAUNCHER.replace("__SRC__", repr(str(src_dir))).replace("__CODE__", repr(str(code_dir)))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


class Inputs:
    pass


def prepare_inputs():
    I = Inputs()
    I.comp = find_mount(COMP, "test.csv")
    I.ds = find_mount(DS_SLUG, "checkpoint-manifest.json")
    I.helios = find_mount(HELIOS_SLUG, ANATOMY_FILE)
    log("competition root", I.comp, "| weights", I.ds, "| heliosli", I.helios)
    man_path = I.ds / "code_manifest.json"
    I.manifest_sha256 = sha256_file(man_path)
    man = json.loads(man_path.read_text())
    man["payload_members"] = man["members"]
    if sha256_file(I.ds / B_MODEL["file"]) != B_MODEL["sha256"]:
        raise K1Error("model sha differs from the kernel constant")
    if sha256_file(I.helios / ANATOMY_FILE) != ANATOMY_SHA256:
        raise K1Error("anatomy.pt differs from the S6 manifest anchor checkpoint")
    cfg = json.loads((I.ds / B_MODEL["run_config"]).read_text())
    want = {"size": B_GEOMETRY["size"], "crop": B_GEOMETRY["crop"], "views": B_GEOMETRY["views"], "slices": B_GEOMETRY["slices"],
            "phys_mm": B_GEOMETRY["phys_mm"], "backbone": B_GEOMETRY["backbone"], "spatial_attn": True, "arms": B_MODEL["arm"]}
    bad = {k: (cfg.get(k), v) for k, v in want.items() if cfg.get(k) != v}
    if bad:
        raise K1Error("run_config differs from the inference arguments (trained, inferred): %s" % bad)
    if B_MODEL["done"] is not None:
        done = json.loads((I.ds / B_MODEL["done"]).read_text())
        if done.get("model_sha256") != B_MODEL["sha256"] or os.path.basename(done.get("model", "")) != B_MODEL.get("source_file", B_MODEL["file"]):
            raise K1Error("DONE marker does not record this model")
        for k in ("epochs", "train_studies"):
            if B_MODEL.get("expect_" + k) is not None and done.get(k) != B_MODEL["expect_" + k]:
                raise K1Error("DONE marker %s=%s, expected %s" % (k, done.get(k), B_MODEL["expect_" + k]))
    I.code_root = TMP / "k1src"
    # code = the scored run's exact file set: the extras from our dataset, everything else straight from Heliosli's; every
    # file is checked against the scored sha256/bytes, so the tree is byte-identical to the one that scored 0.950
    extras = {n: man["members"][n] for n in man["from_this_dataset"]}
    extract_payload(I.ds / EXTRAS_FILE, I.code_root, extras)
    for n, rec in man["members"].items():
        if n in extras:
            continue
        src, dst = I.helios / safe_rel(n), I.code_root / safe_rel(n)
        if not src.is_file() or src.stat().st_size != rec["bytes"]:
            raise K1Error("heliosli file %s missing or size differs" % n)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        if sha256_file(dst) != rec["sha256"]:
            raise K1Error("heliosli file %s sha differs" % n)
    I.src = I.code_root / "heliosli"
    I.code = I.code_root / "cloud_code"
    for n, want_sha in SOURCE_SHA.items():
        if sha256_file(I.src / n) != want_sha:
            raise K1Error("heliosli/%s differs from SOURCE_SHA" % n)
    for n, want_sha in CODE_SHA.items():
        if sha256_file(I.code / "training" / n) != want_sha:
            raise K1Error("cloud_code/training/%s differs from CODE_SHA" % n)
    I.models = TMP / "k1models"
    if I.models.exists():
        shutil.rmtree(I.models)
    I.models.mkdir(parents=True)
    os.symlink(I.ds / B_MODEL["file"], I.models / B_MODEL["file"])        # the models folder holds exactly this one file
    I.anatomy = I.helios / ANATOMY_FILE
    I.launcher = write_launcher(I.src, I.code, TMP / "k1launch" / "k1_b_infer.py")
    test = pd.read_csv(I.comp / "test.csv", dtype={"StudyInstanceUID": str})
    I.test_ids = test["StudyInstanceUID"].astype(str).tolist()
    if not I.test_ids or len(I.test_ids) != len(set(I.test_ids)):
        raise K1Error("test.csv empty or has duplicate study ids")
    ss = I.comp / "sample_submission.csv"
    I.sub_cols = list(pd.read_csv(ss, nrows=1).columns) if ss.is_file() else ["StudyInstanceUID"] + LAB
    if I.sub_cols[0] != "StudyInstanceUID" or sorted(I.sub_cols[1:]) != sorted(LAB):
        raise K1Error("submission columns differ from the 12 findings: %s" % I.sub_cols)
    I.cost = {}
    for split in ("test", "train") if MODE == "parity" else ("test",):
        sc = pd.read_csv(I.comp / (split + "_series.csv"), dtype=str, usecols=["StudyInstanceUID"])
        I.cost.update(sc["StudyInstanceUID"].value_counts().to_dict())
    log("inputs ok: %d test studies | MANIFEST %s | model %s %s... | %d payload files" % (
        len(I.test_ids), I.manifest_sha256[:16], B_MODEL["file"], B_MODEL["sha256"][:16], len(man["payload_members"])))
    _RECEIPT["inputs"] = {"comp": str(I.comp), "ds": str(I.ds), "manifest_sha256": I.manifest_sha256, "model": B_MODEL,
                          "n_test": len(I.test_ids), "sub_cols": I.sub_cols}
    return I


# ---- component B --------------------------------------------------------------------------------------------------------------
def b_cmd(I, split, assignment, out_dir, tmp_dir, deadline):
    g = B_GEOMETRY
    return [sys.executable, "-u", str(I.launcher), "--data", str(I.comp), "--split", split, "--code", str(I.code),
            "--models", str(I.models), "--arm", B_MODEL["arm"], "--backbone", g["backbone"], "--size", str(g["size"]),
            "--crop", str(g["crop"]), "--views", g["views"], "--slices", str(g["slices"]), "--phys-mm", str(g["phys_mm"]),
            "--eval-stride", str(g["eval_stride"]), "--anchor-weights", str(I.anatomy), "--amp", B_AMP, "--chunk", str(g["chunk"]),
            "--workers", str(B_WORKERS), "--item-timeout", str(g["item_timeout"]), "--assignment", str(assignment),
            "--output", str(out_dir), "--tmp", str(tmp_dir), "--dump-logits", str(Path(out_dir) / "logits.json"),
            "--deadline", "%.0f" % deadline]


def split_parts(ids, cost, n):
    order = sorted(ids, key=lambda u: (-cost.get(u, 0), u))
    parts = [order[i::n] for i in range(n)]
    return [p[::2] + p[1::2] for p in parts if p]


def b_env(gi, px_dir=None):
    visible = [d.strip() for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d.strip()]
    dev = visible[gi] if gi < len(visible) else str(gi)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=dev, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MALLOC_ARENA_MAX="2", PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1")
    env.pop("K1_PX_DUMP", None)
    if px_dir is not None:
        env["K1_PX_DUMP"] = str(px_dir)
    return env


def collect_part(pdir, part, I):
    rec = json.loads((pdir / "receipt.json").read_text())
    if rec.get("status") != "PASS" or rec.get("studies") != len(part):
        raise K1Error("%s: receipt status %s, studies %s (expected %d)" % (pdir.name, rec.get("status"), rec.get("studies"), len(part)))
    if rec.get("amp") != B_AMP or rec.get("models") != [B_MODEL["file"]] or rec.get("spatial_models") != 1:
        raise K1Error("%s: receipt amp %s models %s spatial %s" % (pdir.name, rec.get("amp"), rec.get("models"), rec.get("spatial_models")))
    if rec.get("anchor_weights_sha256") != ANATOMY_SHA256:
        raise K1Error("%s: anchor weights sha differs" % pdir.name)
    if not isinstance(rec.get("timed_out"), bool):           # compact_infer always states it (True / False)
        raise K1Error("%s: the receipt does not state timed_out (%r)" % (pdir.name, rec.get("timed_out")))
    shas = rec.get("sha256")
    if not isinstance(shas, dict) or not set(B_RECEIPT_SOURCES) <= set(shas):
        raise K1Error("%s: the receipt's source shas are incomplete: %s" % (pdir.name, sorted(shas) if isinstance(shas, dict) else shas))
    for n, s in shas.items():
        if SOURCE_SHA.get(n) != s:
            raise K1Error("%s: source %s sha differs in the receipt" % (pdir.name, n))
    n_anchor = sum(int(v) for v in (rec.get("anchor_methods") or {}).values())
    if n_anchor != rec.get("scored"):                  # every scored study must have had its SKM-TEA anchor record (review M1)
        raise K1Error("%s: %s scored studies but %d anchor records %s" % (pdir.name, rec.get("scored"), n_anchor, rec.get("anchor_methods")))
    logits = json.loads((pdir / "logits.json").read_text())
    pc = pd.read_csv(pdir / "predictions.csv", dtype={"StudyInstanceUID": str})
    if list(pc.columns) != ["StudyInstanceUID"] + LAB:
        raise K1Error("%s: predictions.csv columns %s" % (pdir.name, list(pc.columns)))
    pc = pc.set_index("StudyInstanceUID")
    z = {}
    for u in part:
        if u not in logits:
            continue
        a = np.asarray(logits[u], np.float64)
        if a.shape != (1, len(LAB)) or not np.isfinite(a).all():
            raise K1Error("%s: logits for %s have shape %s or are non-finite" % (pdir.name, u, a.shape))
        if u not in pc.index:
            raise K1Error("%s: %s has logits but no predictions.csv row" % (pdir.name, u))
        zm = a.mean(0)
        p = 1.0 / (1.0 + np.exp(-zm))
        d = float(np.abs(p - pc.loc[u, LAB].to_numpy(np.float64)).max())
        if d >= 1e-5:
            raise K1Error("%s: predictions.csv disagrees with the logits for %s (%.2e)" % (pdir.name, u, d))
        z[u] = zm
    extra = set(logits) - set(part)
    if extra:
        raise K1Error("%s: logits for studies outside the assignment: %s" % (pdir.name, sorted(extra)[:3]))
    fallback = dict(rec.get("fallback") or {})
    missing = [u for u in part if u not in z]
    for u in missing:
        fallback.setdefault(u, "no logits")
    return z, fallback, rec


def progress_lines(logp):
    out = []
    for l in tail(logp, 10 ** 7).splitlines():
        if l.startswith("{") and '"done"' in l:
            try:
                out.append(json.loads(l))
            except Exception:
                pass
    return out


def b_fail_cap(n):
    return max(B_FAIL_MIN, math.ceil(B_FAIL_FRAC * n))


def b_done(pdir):
    return (pdir / "receipt.json").is_file() and (pdir / "logits.json").is_file() and "WORKER_DONE" in tail(pdir / "log.txt", 3000)


def run_b(I, tag, ids, split, budget_s, px=False):
    """B on ids over one process per GPU, one retry of failed studies.  Returns (P, Z, info, failed): failed = {uid: reason} of
    studies still without a prediction (their P / Z rows are NaN); at most b_fail_cap(N) when tag == "B", else none.  A deadline
    (first pass or retry) is always a HARD ERROR: a receipt whose timed_out is not False, or any fallback reason naming the
    deadline.  px: every item's pixel / meta hashes are merged into <tag>/px_hashes.json.  info_sha256 (returned, so it reaches
    k1_receipt.json) binds <tag>_info.json, which compare_parity reads for the amp and runtime checks."""
    def deadline_named(fb):        # local, not module level: Astra's probe lifts run_b alone.  compact_infer's reason for a study
        return [u for u, v in fb.items() if "deadline" in str(v).lower()]     # the deadline cut off: "not reached before the deadline"
    root = OUT / tag
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    n_gpu = max(1, int(_RECEIPT["env"].get("n_gpu", 1)))
    parts = split_parts(ids, I.cost, n_gpu)
    t_start = time.time()
    deadline = t_start + max(budget_s - B_DEADLINE_MARGIN_S, budget_s * 0.8)
    specs, pdirs = [], []
    for gi, part in enumerate(parts):
        pdir = root / ("part%d" % gi)
        pdir.mkdir()
        (pdir / "ids.json").write_text(json.dumps(part))
        px_dir = None
        if px:
            px_dir = pdir / "px"; px_dir.mkdir()
        specs.append({"name": "%s.part%d" % (tag, gi), "cmd": b_cmd(I, split, pdir / "ids.json", pdir, TMP / ("%s_tmp%d" % (tag, gi)), deadline),
                      "env": b_env(gi, px_dir), "cwd": str(TMP), "log": pdir / "log.txt", "done_when": (lambda d=pdir: b_done(d))})
        pdirs.append(pdir)
    (root / "argv_part0.json").write_text(json.dumps(specs[0]["cmd"]))
    log("[%s] %d studies over %d GPU process(es) %s, budget %ds, compact_infer deadline +%ds" % (
        tag, len(ids), len(parts), [len(p) for p in parts], budget_s, deadline - t_start))
    runs = run_group(specs, budget_s, tag)
    z, fallback, receipts = {}, {}, []
    for pdir, part in zip(pdirs, parts):
        zz, fb, rec = collect_part(pdir, part, I)
        z.update(zz); fallback.update(fb)
        receipts.append({k: rec.get(k) for k in ("studies", "scored", "seconds", "amp", "gpu", "torch", "timm", "stats", "timed_out",
                                                 "anchor_methods", "per_study")})
        receipts[-1]["progress"] = progress_lines(pdir / "log.txt")
    timed = [i for i, rc in enumerate(receipts) if rc.get("timed_out") is not False]
    if timed or deadline_named(fallback):
        # the deadline came first: a retry cannot reach them either (review m4); no allowance ever covers a deadline.  A receipt
        # that does not state timed_out False counts as one (collect_part already refuses it; this holds even without it)
        raise K1Error("%s: compact_infer reached its deadline (budget %ds; timed_out not False in part(s) %s); %d studies have no "
                      "prediction: %s" % (tag, budget_s, timed, len(fallback), dict(list(fallback.items())[:5])))
    retry = None
    if fallback:
        left = t_start + budget_s - time.time()
        log("[%s] %d fallback studies %s; %.0fs of budget left" % (tag, len(fallback), dict(list(fallback.items())[:5]), left))
        if left < B_RETRY_MIN_S:
            raise K1Error("%s: %d studies failed and no time is left for the retry: %s" % (tag, len(fallback), dict(list(fallback.items())[:10])))
        rdir = root / "retry"
        rdir.mkdir()
        rids = sorted(fallback)
        (rdir / "ids.json").write_text(json.dumps(rids))
        rdeadline = time.time() + max(left - B_DEADLINE_MARGIN_S, left * 0.8)
        if px:
            (rdir / "px").mkdir()
        rruns = run_group([{"name": tag + ".retry", "cmd": b_cmd(I, split, rdir / "ids.json", rdir, TMP / (tag + "_tmp_retry"), rdeadline),
                            "env": b_env(0, (rdir / "px") if px else None), "cwd": str(TMP), "log": rdir / "log.txt",
                            "done_when": lambda: b_done(rdir)}], left, tag)
        zz, fb2, rec2 = collect_part(rdir, rids, I)
        if rec2.get("timed_out") is not False or deadline_named(fb2):
            # review C1: a retry that reaches its deadline is a HARD ERROR (PREREG section 5.1), never a capped B failure: the
            # allowance below covers only studies that still fail after a retry that completed within its deadline
            raise K1Error("%s: compact_infer reached its deadline in the retry (%.0fs left at its start; timed_out %r); %d of %d "
                          "retried studies have no prediction: %s" % (tag, left, rec2.get("timed_out"), len(fb2), len(rids),
                                                                       dict(list(fb2.items())[:5])))
        z.update(zz)
        retry = {"ids": rids, "first_reasons": fallback, "still_failing": fb2, "seconds": rec2.get("seconds"),
                 "timed_out": rec2.get("timed_out"), "runs": rruns, "anchor_methods": rec2.get("anchor_methods")}
        fallback = fb2
    missing = [u for u in ids if u not in z]
    failed = {u: fallback.get(u, "no logits") for u in missing}
    cap = b_fail_cap(len(ids)) if tag == "B" else 0
    if len(failed) > cap:
        raise K1Error("%s: %d of %d studies have no B prediction after the retry (allowed %d): %s" % (
            tag, len(failed), len(ids), cap, dict(list(failed.items())[:10])))
    nan12 = np.full(len(LAB), np.nan)
    Z = np.stack([z.get(u, nan12) for u in ids])
    P = 1.0 / (1.0 + np.exp(-Z))
    ok = np.array([u in z for u in ids])
    if not np.isfinite(P[ok]).all():
        raise K1Error("%s: non-finite B probabilities" % tag)
    df = pd.DataFrame({"StudyInstanceUID": ids})
    for j, l in enumerate(LAB):
        df[l] = P[:, j]
    for j, l in enumerate(LAB):
        df["z|" + l] = Z[:, j]
    df.to_csv(root / ("%s_probs.csv" % tag), index=False, float_format="%.17g")
    info = {"seconds": round(time.time() - t_start, 1), "budget_s": round(budget_s), "parts": [len(p) for p in parts], "runs": runs,
            "receipts": receipts, "retry": retry, "failed_after_retry": failed, "fail_cap": cap,
            "probs_sha256": sha256_file(root / ("%s_probs.csv" % tag))}
    if px:                                    # pixel evidence (review C3): one record per study; a retried study's retry record wins
        recs, bad, clean = {}, [], lambda r: r.get("error") is None and not r.get("hash_error")
        for f in sorted(root.glob("*/px/px_*.jsonl")):                     # part0, part1, ..., retry: sorted, so the retry is last
            for k, line in enumerate(f.read_text(errors="replace").splitlines()):
                try:
                    r = json.loads(line)
                except ValueError:
                    r = None
                if not isinstance(r, dict) or not isinstance(r.get("uid"), str):
                    # a truncated line (a worker killed mid-write): diagnostic evidence never hard-fails the stage; the study is
                    # then without a record unless another line covers it, and compare_parity FAILs missing evidence
                    bad.append("%s:%d" % (f.relative_to(root), k + 1))
                    continue
                old = recs.get(r["uid"])
                if old is None or clean(r) or not clean(old):    # a clean record is never replaced by an errored duplicate (a
                    recs[r["uid"]] = r                           # loader restart can re-dump an item in another worker's file)
        if bad:
            log("[%s] px: %d unreadable line(s) skipped: %s" % (tag, len(bad), bad[:5]))
        save_json(root / "px_hashes.json", recs)
        info["px_hashes_sha256"] = sha256_file(root / "px_hashes.json")
        info["px_studies"] = len(recs)
        info["px_unreadable_lines"] = {"count": len(bad), "first": bad[:20]}
    save_json(root / ("%s_info.json" % tag), info)
    info["info_sha256"] = sha256_file(root / ("%s_info.json" % tag))   # the file holds everything above; the receipt binds it
    log("[%s] ok: %d studies, %.0fs, retry %s, failed after retry %d (cap %d)" % (
        tag, len(ids), info["seconds"], None if retry is None else len(retry["ids"]), len(failed), cap))
    return P, Z, info, failed


# ---- blend --------------------------------------------------------------------------------------------------------------------
def rank01(P):
    """Per column: average-tie ranks 1..n divided by n, float64 (= scipy.stats.rankdata(method='average') / n, gate_score.rank01)."""
    P = np.asarray(P, np.float64)
    n = P.shape[0]
    out = np.empty(P.shape, np.float64)
    for j in range(P.shape[1]):
        v = P[:, j]
        o = np.argsort(v, kind="mergesort")
        s = v[o]
        first = np.r_[True, s[1:] != s[:-1]]
        starts = np.flatnonzero(first)
        ends = np.r_[starts[1:], n] - 1
        avg = (starts + ends) / 2.0 + 1.0
        r = np.empty(n, np.float64)
        r[o] = avg[np.cumsum(first) - 1]
        out[:, j] = r / n
    return out


def write_submission(ids, sub_cols, values):
    sub = pd.DataFrame(np.asarray(values).astype(np.float32), columns=LAB)
    sub.insert(0, "StudyInstanceUID", ids)
    sub = sub[sub_cols]
    if list(sub.columns) != sub_cols or sub["StudyInstanceUID"].tolist() != list(ids):
        raise K1Error("submission column/row identity drift")
    v = sub[LAB].to_numpy(np.float64)
    if not np.isfinite(v).all() or (v < 0).any() or (v > 1).any():
        raise K1Error("submission has non-finite or out-of-range values")
    tmp = WORK / ".submission.csv.tmp"
    sub.to_csv(tmp, index=False)
    os.replace(tmp, WORK / "submission.csv")
    return WORK / "submission.csv"


def blend_with_failures(B, failed_mask):
    """rank01(B); a study B failed on is filled with the column mean of the scored rows before ranking (Heliosli's controller
    fill, the original's B side), 0.5 when no row was scored."""
    failed_mask = np.asarray(failed_mask, bool)
    Bf = np.array(B, np.float64)
    if failed_mask.any():
        Bf[failed_mask] = np.nanmean(Bf[~failed_mask], axis=0) if (~failed_mask).any() else 0.5
    return rank01(Bf)


def blend_and_write(I, B, b_failed=None):
    if B.shape != (len(I.test_ids), len(LAB)):
        raise K1Error("component B shape %s" % (B.shape,))
    b_failed = dict(b_failed or {})
    fmask = np.array([u in b_failed for u in I.test_ids])
    ok = ~fmask
    if not np.isfinite(B[ok]).all():
        raise K1Error("B is non-finite on a scored row")
    final = blend_with_failures(B, fmask)
    if not np.isfinite(final).all():
        raise K1Error("blend is non-finite")
    info = {"model": B_MODEL["file"], "arm": B_MODEL["arm"], "rows": len(I.test_ids), "b_failed_mean_fill": b_failed}
    p = write_submission(I.test_ids, I.sub_cols, final)
    info["submission_sha256"] = sha256_file(p)
    save_json(OUT / "blend_receipt.json", info)
    log("[blend] wrote %s | %d rows | B failed (column-mean fill) %d | sha %s..." % (
        p, len(I.test_ids), len(b_failed), info["submission_sha256"][:16]))
    return info


# ---- parity ------------------------------------------------------------------------------------------------------------------
def run_parity(I):
    P, Z, info, _ = run_b(I, "parity", list(PARITY_IDS), "train", PARITY_TIMEOUT_S, px=True)
    px = json.loads((OUT / "parity" / "px_hashes.json").read_text())           # merged by run_b
    for u, z in zip(PARITY_IDS, Z):
        log("K1_PARITY_Z %s %s" % (u, " ".join("%.6f" % x for x in z)))
    for u in PARITY_IDS:
        r = px.get(u, {})
        log("K1_PARITY_PX %s n=%s img=%s meta=%s anchor=%s" % (u, r.get("n"), str(r.get("img_sha256"))[:16],
                                                               str(r.get("meta_sha256"))[:16], (r.get("anchor") or {}).get("method")))
    timing = [{"part": i, "studies": rc["studies"], "seconds": rc["seconds"], "progress": rc["progress"]}
              for i, rc in enumerate(info["receipts"])]
    log("K1_PARITY_TIMING " + json.dumps(timing))
    return info


# ---- main ---------------------------------------------------------------------------------------------------------------------
def main():
    global _LOG
    OUT.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    remove_submission()
    for stale in ("k1_error.json", "k1_receipt.json", "blend_receipt.json"):
        try:
            (OUT / stale).unlink()
        except FileNotFoundError:
            pass
    _LOG = open(OUT / "k1_log.txt", "a")
    errors = []
    try:
        _RECEIPT["self"] = self_identity()
        log("K1_BUILD %s" % json.dumps({"build_id": K1_BUILD_ID, "run": _RUN, "self": _RECEIPT["self"]}))
        arm_guard(GUARD_S)
        log("leg_s6 mode %s | python %s | model %s %s... arm %s | budgets B floor %ds parity %ds guard %ds" % (
            MODE, sys.version.split()[0], B_MODEL["file"], B_MODEL["sha256"][:12], B_MODEL["arm"], B_TIMEOUT_S,
            PARITY_TIMEOUT_S if MODE == "parity" else 0, GUARD_S))
        stage("env")
        env = probe_env()
        _RECEIPT["env"] = env
        log("K1_ENV", json.dumps(env))
        if not env.get("cuda_available") or env.get("n_gpu", 0) < 1:
            raise K1Error("CUDA is not available")
        stage_done("env")
        stage("inputs")
        I = prepare_inputs()
        stage_done("inputs")
        stage("B_probe")
        _RECEIPT["B_probe"] = probe_b_stack(I)
        log("K1_BPROBE", json.dumps(_RECEIPT["B_probe"]))
        stage_done("B_probe")
        if MODE == "parity":
            stage("parity")
            try:
                _RECEIPT["parity"] = run_parity(I)
                stage_done("parity", ok=True)
            except Exception as e:                       # parity is a diagnostic: record, run the main path, fail at the end
                errors.append("parity: %r" % e)
                log("[parity] FAILED:", repr(e)[:3000])
                stage_done("parity", ok=False, error=repr(e)[:3000])
        stage("B")
        b_budget = guard_left() - END_SLACK_S          # B gets everything the guard has left (>= its floor B_TIMEOUT_S)
        if b_budget < B_TIMEOUT_S:
            log("[B] WARNING: only %ds of guard time left, below the B floor %ds" % (b_budget, B_TIMEOUT_S))
        # parity mode: B's pixel / meta hashes on the 3 public studies too (the only test-split evidence; review C3)
        B, _, b_info, b_failed = run_b(I, "B", I.test_ids, "test", b_budget, px=(MODE == "parity"))
        _RECEIPT["B"] = {k: v for k, v in b_info.items() if k != "receipts"}
        stage_done("B")
        stage("blend")
        _RECEIPT["blend"] = blend_and_write(I, B, b_failed)
        stage_done("blend")
        if errors:
            raise K1Error("; ".join(errors))
        _RECEIPT["total_seconds"] = round(time.time() - T0, 1)
        save_json(OUT / "k1_receipt.json", _RECEIPT)
        faulthandler.cancel_dump_traceback_later()
        _GUARD_STOP.set()
        log("K1_DONE %s" % json.dumps({"mode": MODE, "rows": len(I.test_ids), "seconds": _RECEIPT["total_seconds"],
                                       "stages": {k: v.get("seconds") for k, v in _RECEIPT["stages"].items()},
                                       "submission_sha256": _RECEIPT["blend"]["submission_sha256"]}))
    except BaseException as e:
        kill_all_children()
        remove_submission()
        _RECEIPT["total_seconds"] = round(time.time() - T0, 1)
        try:
            save_json(OUT / "k1_error.json", {"error": repr(e)[:5000], "traceback": traceback.format_exc()[-5000:], "receipt": _RECEIPT})
        except Exception:
            pass
        log("K1_HARD_ERROR (submission.csv deleted): %r" % (e,))
        raise


if __name__ == "__main__":
    main()
