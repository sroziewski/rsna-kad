from __future__ import annotations
import time as _startup_time
_STARTUP_T0 = _startup_time.perf_counter()
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '4')
import gc
import hashlib
import json
import re
import time
import traceback
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
import torch.nn.functional as F

def _cuda_execution_probe(index):
    dev = torch.device(f'cuda:{index}')
    try:
        major, minor = torch.cuda.get_device_capability(index)
        probe = nn.Conv2d(3, 4, kernel_size=3, padding=1).eval().to(dev)
        with torch.inference_mode():
            out = probe(torch.zeros((1, 3, 16, 16), device=dev))
            if tuple(out.shape) != (1, 4, 16, 16):
                raise RuntimeError(f'unexpected CUDA probe shape {tuple(out.shape)}')
        torch.cuda.synchronize(index)
        print(f'cuda:{index} probe PASS (compute {major}.{minor})')
        del probe, out
        torch.cuda.empty_cache()
        return True
    except Exception as exc:
        print(f'cuda:{index} probe FAIL ({type(exc).__name__}: {exc}); rejecting device')
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        return False
DEVS = []
if torch.cuda.is_available():
    DEVS = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count()) if _cuda_execution_probe(i)]
if len(DEVS) != 2:
    raise RuntimeError('This complete ensemble requires two working GPUs; no CPU/single-GPU partial fallback')
if any('T4' not in torch.cuda.get_device_name(d) for d in DEVS):
    raise RuntimeError('Select the reference T4 x2 accelerator')
print(f'devices: {[str(d) for d in DEVS]}')
_STARTUP_GPU_IMPORT_S = _startup_time.perf_counter() - _STARTUP_T0
print(f'[startup] imports+GPU preflight: {_STARTUP_GPU_IMPORT_S:.2f}s', flush=True)
T0 = time.time()
SEED = 2026
np.random.seed(SEED)
torch.manual_seed(SEED)
TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
CROP_MM = 130.0
CACHE_IMG = 336
GROUP = 3
N_GROUP_MAX = 1
CACHE_FRACTION = 0.45
CACHE_BUDGET_MAX_GB = 24.0
CACHE_BUDGET_GB = 12.0
TEST_SHARE = 0.3
HDR_THREADS = 16
PIX_THREADS = 12
ORDER_THREADS = 32
ORDER_BUDGET_S = 5400
RUNS = [{'name': 'r224', 'img': 224}, {'name': 'r336', 'img': 336}]
EPOCHS = 10
BATCH_STUDIES = 8
AUG_ROT_DEG = 8.0
AUG_SCALE = 0.08
AUG_SHIFT = 0.05
AUG_INTENSITY = 0.1
LAT_MIN_OFFSET_MM = 20.0
SLICE_BAND = (0.2, 0.8)
RULES_NATIVE = {'order': 'normal', 'lat': 'centre', 'slot_fallback': False, 'decode_fill': 'nearest'}
RULES_LEGACY = {'order': 'dominant_axis', 'lat': 'corner_x', 'slot_fallback': True, 'decode_fill': 'zero'}
RULES = dict(RULES_NATIVE)
LEGACY_LAT_OFFSET_MM = 5.0
LR_HEAD = 0.001
LR_BACKBONE = 8e-06
UNFREEZE_LAST = 6
WEIGHT_DECAY = 0.02
EVAL_BATCH = 8
TIME_BUDGET = 8.0 * 3600
SLOTS_RECOVERED = [('SAG_FLUID_FS', 'Sagittal', True, True), ('COR_FLUID_FS', 'Coronal', True, True), ('AX_FLUID_FS', 'Axial', True, True), ('SAG_FLUID_NOFS', 'Sagittal', True, False), ('COR_T1', 'Coronal', False, False), ('SAG_T1', 'Sagittal', False, False)]
SLOTS_PUBLIC = [('SAG_FLUID', 'Sagittal', None, True), ('COR_FLUID', 'Coronal', None, True), ('AX_FLUID', 'Axial', None, True), ('SAG_STRUCT', 'Sagittal', None, False), ('COR_STRUCT', 'Coronal', None, False), ('AX_STRUCT', 'Axial', None, False)]
SLOT_SCHEME = os.environ.get('SLOT_SCHEME', 'recovered')
SLOTS = SLOTS_PUBLIC if SLOT_SCHEME == 'public' else SLOTS_RECOVERED
N_SLOT = len(SLOTS)
POOL_PARTS = {'cls_mean': 2, 'cls_mean_focal': 3}
SLOT_PRIOR_TABLE = {'ACL': (0, 3, 5), 'MCL': (1, 4), 'Medial Meniscus': (0, 1, 3, 4), 'Lateral Meniscus': (0, 1, 3, 4), 'Medial OA': (1, 4, 5), 'Lateral OA': (1, 4, 5), 'PF OA': (0, 2, 5), 'Effusion': (0, 2), 'Synovitis': (0, 2), "Baker's": (0,), 'Contusion': (0, 1, 2), 'Fracture': (0, 1, 2, 4, 5)}
SLOT_PRIOR_STRENGTH = 0.55
FATSAT_OPTS = {'FS', 'FATSAT', 'FAT_SAT', 'FSAT'}
_SEP = re.compile('[_\\-.]')
_FATSAT_RX = re.compile('\\bfs\\b|fatsat|fat sat|\\bstir\\b|\\bspair\\b|\\bspir\\b|\\bwe\\b|water excit|\\btirm\\b|\\bsting\\b|\\bfatsup\\b|smart fat|\\bwater\\b')  # 2026-09-17: GE Dixon water images ('SMART FAT', 'Water: SMART FAT', seq FSEfw) are fat-suppressed; 37/4407 training studies had no fat-sat series detected before
_T1_RX = re.compile('\\bt1\\b|\\bt1w\\b')
_T2_RX = re.compile('\\bt2\\b|\\bt2w\\b')
_PD_RX = re.compile('\\bpd\\b|\\bpdw\\b|proton|\\bdp\\b|dens')

# Runtime integrity: no partial ensemble is ever published as submission.csv.
import os, json, hashlib, time, threading, tempfile
from pathlib import Path
import numpy as np
import pandas as pd

_RSNA_AUDIT_LOCK = threading.RLock()
_RSNA_AUDIT = {'contract': 'btkd_speedy_v558_input_uniform', 'events': [], 'phases': {}}
_RSNA_ORDER_MEMO = {}
_RSNA_HEADERS_MEMO = {}
_RSNA_CACHE_FILES = []
_RSNA_TEST_IDS = None
_RSNA_LABELS = ['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']

def rsna_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()

def rsna_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + path.name + '.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False, default=str)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

def rsna_event(kind, **details):
    with _RSNA_AUDIT_LOCK:
        _RSNA_AUDIT['events'].append({'kind':kind, **details})

def rsna_finite(values, tag, probability=False):
    x = np.asarray(values)
    if not np.issubdtype(x.dtype, np.number) or not np.isfinite(x).all():
        raise RuntimeError(f'{tag}: nonfinite/non-numeric values BEFORE ranking')
    if probability and x.size and (x.min() < 0 or x.max() > 1):
        raise RuntimeError(f'{tag}: probability outside [0,1]')
    return x

def rsna_rank01(values):
    """Parent endpoint scale; tied values share their average rank."""
    x = np.asarray(rsna_finite(values, 'rank01'), dtype=np.float64)
    if x.ndim != 2 or not len(x):
        raise ValueError('rank01 requires a nonempty [study,finding] matrix')
    if len(x) == 1:
        return np.full_like(x, .5)
    return (pd.DataFrame(x).rank(method='average').to_numpy() - 1) / (len(x)-1)

def rsna_rankpct(values):
    x = np.asarray(rsna_finite(values, 'rankpct'), dtype=np.float64)
    if x.ndim != 2 or not len(x):
        raise ValueError('rankpct requires a nonempty matrix')
    return pd.DataFrame(x).rank(method='average', pct=True).to_numpy(np.float64)

def rsna_frame(frame, ids, labels, tag):
    ids = [str(u) for u in ids]; labels = list(labels)
    if len(ids) != len(set(ids)):
        raise RuntimeError(f'{tag}: duplicate expected UID')
    if frame.columns.tolist() != ['StudyInstanceUID', *labels]:
        raise RuntimeError(f'{tag}: label order/schema mismatch')
    frame = frame.copy(); frame['StudyInstanceUID'] = frame['StudyInstanceUID'].astype(str)
    if frame.StudyInstanceUID.duplicated().any() or set(frame.StudyInstanceUID) != set(ids):
        raise RuntimeError(f'{tag}: missing/extra/duplicate UID')
    frame = frame.set_index('StudyInstanceUID').loc[ids].reset_index()
    rsna_finite(frame[labels].to_numpy(), tag, probability=True)
    return frame

def rsna_save_predictions(tag, ids, values, labels=None):
    x = rsna_finite(values, tag)
    target = Path('/kaggle/working/diagnostics'); target.mkdir(parents=True, exist_ok=True)
    path = target/(tag+'.npz')
    np.savez_compressed(path, study_uids=np.asarray(ids,dtype=str), labels=np.asarray(labels or _RSNA_LABELS,dtype=str), values=x)
    with _RSNA_AUDIT_LOCK:
        _RSNA_AUDIT['phases'][tag]={'path':str(path),'sha256':rsna_sha(path),'shape':list(x.shape)}

def rsna_strict_load(model, state, tag):
    """A random frozen encoder is never an acceptable missing-state fallback."""
    expected = model.state_dict()
    missing = sorted(set(expected)-set(state)); extra = sorted(set(state)-set(expected))
    bad_shape = [k for k in set(expected)&set(state) if tuple(expected[k].shape)!=tuple(state[k].shape)]
    if missing or extra or bad_shape:
        raise RuntimeError(f'{tag}: incomplete/mismatched checkpoint; missing={missing[:20]}, extra={extra[:20]}, shapes={bad_shape[:20]}. A partial encoder needs its exact pinned source weights; random initialization is forbidden.')
    for k,t in state.items():
        if t.is_floating_point() and not bool(t.isfinite().all()):
            raise RuntimeError(f'{tag}: nonfinite checkpoint tensor {k}')
    return model.load_state_dict(state, strict=True)

def rsna_deadline(tag):
    if time.time() - T0 > TIME_BUDGET:
        raise TimeoutError(f'{tag}: full-run budget exceeded; no incomplete output may be submitted')

def rsna_array(shape, tag):
    """Large temporary pixels belong in scratch space, not saved notebook outputs."""
    size = int(np.prod(shape))
    if size <= float(os.environ.get('RSNA_CACHE_RAM_GIB', '1')) * (1 << 30):
        return np.zeros(shape, np.uint8)
    directory = Path(os.environ.get('RSNA_PIXEL_SCRATCH', '/kaggle/temp/rsna_pixels_v559'))
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as scratch_error:  # /kaggle/temp is not guaranteed; /tmp is non-persisted scratch with ~1 TiB free
        rsna_event('scratch_fallback', tag=tag, requested=str(directory), error=f'{type(scratch_error).__name__}: {scratch_error}')
        directory = Path('/tmp/rsna_pixels_v559')
        directory.mkdir(parents=True, exist_ok=True)
    import shutil
    free = shutil.disk_usage(directory).free
    rsna_event('scratch_allocation', tag=tag, bytes=size, directory=str(directory), free_bytes=int(free))
    if free < size + (1 << 30) and str(directory) != '/tmp/rsna_pixels_v559':
        rsna_event('scratch_fallback', tag=tag, requested=str(directory), error=f'insufficient free space ({free} bytes)')
        directory = Path('/tmp/rsna_pixels_v559'); directory.mkdir(parents=True, exist_ok=True); free = shutil.disk_usage(directory).free
    if free < size + (1 << 30):
        raise RuntimeError(f'{tag}: not enough scratch disk for complete cache ({size} bytes; {free} free)')
    fd, path = tempfile.mkstemp(prefix='pixels_', suffix='.npy', dir=directory)
    os.close(fd)
    try:
        a = np.lib.format.open_memmap(path, mode='w+', dtype=np.uint8, shape=tuple(shape))
        a[:] = 0
    except BaseException:
        Path(path).unlink(missing_ok=True)
        raise
    _RSNA_CACHE_FILES.append(path)
    return a

def rsna_release_pixels(array):
    if isinstance(array,np.memmap):
        path=str(array.filename)
        array.flush()
        # POSIX unlink releases the file after the last view is closed.
        Path(path).unlink(missing_ok=True)
        if path in _RSNA_CACHE_FILES: _RSNA_CACHE_FILES.remove(path)

def rsna_initialize(ids):
    global _RSNA_TEST_IDS
    _RSNA_TEST_IDS=[str(u) for u in ids]
    if not _RSNA_TEST_IDS or len(set(_RSNA_TEST_IDS))!=len(_RSNA_TEST_IDS):
        raise RuntimeError('empty or duplicate competition study IDs')
    work=Path('/kaggle/working'); work.mkdir(parents=True,exist_ok=True)
    os.chdir(work)
    for f in ['submission.csv','_pipeline_stage.csv','btkd_v559_complete.json','btkd_v559_complete.json']:
        (work/f).unlink(missing_ok=True)
    (work/'diagnostics').mkdir(exist_ok=True)
    _RSNA_AUDIT['cohort']=len(_RSNA_TEST_IDS)
    _RSNA_AUDIT['environment']={'python':__import__('sys').version,'torch':str(torch.__version__),'cuda':torch.version.cuda,'gpu_names':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}
    rsna_json(work/'diagnostics/runtime_started.json',_RSNA_AUDIT)

def rsna_asset_preflight():
    """Locate required additions before spending time on the public parent."""
    names=['raptor_ft_coatnet_v5_full_swa.pt','raptor_ft_coatnet_v10_full.pt','raptor_ft_coatnet_v8_full_swa.pt']
    receipt={name:str(_asset_find_asset(name)) for name in names}
    for name,digest in [
        ('coat_resgated_ep10_top3_manifest.json','98511a8fdeb9da0e6e70c78d013ff636e1476f31c80b5dc134d294b18c3f284e'),
        ('coatnet_pairfilm_manifest.json','7ada0605bca6b0530569c6454e988ace479606a3328ed591d090e5764fea661d'),
        ('opencv_python_headless-4.12.0.88-*.whl','236c8df54a90f4d02076e6f9c1cc763d794542e886c576a6fee46ec8ff75a7a9')]:
        receipt[name]={'path':str(_asset_find_asset(name,digest)),'sha256':digest}
    roots=[Path(p) for p in ['/kaggle/input/datasets/mattiaangeli/knee-mri-fold-weights','/kaggle/input/knee-mri-fold-weights'] if (Path(p)/'m_f0.pt').is_file()]
    roots=list({str(p.resolve()):p for p in roots}.values())
    if len(roots)!=1:raise RuntimeError(f'A5: expected one complete attached root, found {roots}')
    missing=[str(roots[0]/f'm_f{i}.pt') for i in range(5) if not (roots[0]/f'm_f{i}.pt').is_file()]
    if missing:raise FileNotFoundError(f'A5 missing folds: {missing}')
    receipt['A5_root']=str(roots[0]);receipt['tensor_loading_and_fingerprints']='validated by mandatory component loaders before predictions'
    d4_manifest_path = _asset_find_asset('coatnet_pairfilm_manifest.json',
        '7ada0605bca6b0530569c6454e988ace479606a3328ed591d090e5764fea661d')
    timm_wheel = d4_manifest_path.parent / 'timm-1.0.22-py3-none-any.whl'
    if not timm_wheel.is_file() or rsna_sha(timm_wheel) != '888981753e65cbaacfc07494370138b1700a27b1f0af587f4f9b47bc024161d0':
        raise RuntimeError('D4 pinned timm wheel missing or changed')
    receipt['D4_timm_wheel_sha256'] = rsna_sha(timm_wheel)
    rsna_json('/kaggle/working/diagnostics/asset_preflight.json',receipt)


def rsna_phase(stage, status='START', **extra):
    """Local phase evidence; this does not expose Kaggle's hidden rerun logs."""
    import json, os, resource, time
    from pathlib import Path
    work = Path('/kaggle/working/diagnostics')
    work.mkdir(parents=True, exist_ok=True)
    current = {'stage': stage, 'status': status,
               'elapsed_seconds': time.time()-T0,
               'max_rss_kib': int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss), **extra}
    for path in ['/sys/fs/cgroup/memory.current', '/sys/fs/cgroup/memory.max']:
        try: current[path.rsplit('/', 1)[-1]] = Path(path).read_text().strip()
        except OSError: pass
    if torch.cuda.is_initialized():
        current['gpus'] = [{'id': i, 'allocated_bytes': torch.cuda.memory_allocated(i),
                            'reserved_bytes': torch.cuda.memory_reserved(i)}
                           for i in range(torch.cuda.device_count())]
    print('[phase] '+json.dumps(current, sort_keys=True), flush=True)
    rsna_json(work/'current_phase.json', current)
    with (work/'phase_events.jsonl').open('a') as f:
        f.write(json.dumps(current, sort_keys=True)+'\n')
    return current

# Record uncaught cell errors locally; Kaggle may still withhold hidden-run logs.
def _rsna_capture_cell_error(result):
    error = getattr(result, 'error_in_exec', None) or getattr(result, 'error_before_exec', None)
    if error is None:
        return
    try:
        path = Path('/kaggle/working/diagnostics/current_phase.json')
        previous = json.loads(path.read_text()) if path.is_file() else {}
        rsna_phase(previous.get('stage', 'unknown'), 'FAILED', error_type=type(error).__name__, error=str(error))
    except Exception as logging_error:
        print('[diagnostic-write-failed]', type(logging_error).__name__, flush=True)
try:
    _rsna_ip = get_ipython()
    if _rsna_ip is not None:
        _rsna_ip.events.register('post_run_cell', _rsna_capture_cell_error)
except NameError:
    pass