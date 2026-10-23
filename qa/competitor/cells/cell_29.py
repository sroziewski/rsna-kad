if globals().get('_DINOV2_MATCHED_MEMBERS') != 20:
    raise RuntimeError('DINOv2 20/20 fingerprint gate did not pass')
import gc as _ke_gc
import os as _ke_os
import time as _ke_time
from concurrent.futures import ThreadPoolExecutor as _KeThreadPool
import numpy as _ke_np
import pandas as _ke_pd
from pathlib import Path as _KePath

_ke_primary = _KePath('/kaggle/working/_pipeline_stage.csv')
_ke_ours = _ke_pd.read_csv(_ke_primary, dtype={'StudyInstanceUID': str})
_KE_LAB = [c for c in _ke_ours.columns if c != 'StudyInstanceUID']

_KE_SRC = r'''
import os, glob, time, gc, hashlib
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import timm
torch.backends.cudnn.benchmark = False  # ragged 94-window chunks: no repeated autotuning
torch.backends.cuda.matmul.allow_tf32 = True
IMG = 336
CROP_MM = 140.0
SPAN_LO, SPAN_HI = 0.02, 0.98
SLOTS = [("Sagittal", 1, 18), ("Sagittal", 0, 14),
         ("Coronal", 1, 12), ("Coronal", 0, 8), ("Axial", -1, 12)]
MAXS = sum(slot[2] for slot in SLOTS)
K_EVAL = 62
NORM = "imagenet"
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
       "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
       "Contusion", "Fracture"]
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
_SLOTS64 = [("Sagittal", 1, 18), ("Sagittal", 0, 14),
            ("Coronal", 1, 12), ("Coronal", 0, 8), ("Axial", -1, 12)]
_SLOTS44 = [("Sagittal", 1, 12), ("Sagittal", 0, 10),
            ("Coronal", 1, 8), ("Coronal", 0, 6), ("Axial", -1, 8)]
ARMS = [
    {"name": "maxspan-v5", "file": "raptor_ft_coatnet_v5_full_swa.pt",
     "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384,
     "img": 336, "slots": _SLOTS64, "span": (0.02, 0.98), "k_eval": 62,
     "reverse": False, "w": 0.60},
    {"name": "native384dense-v10", "file": "raptor_ft_coatnet_v10_full.pt",
     "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384,
     "img": 384, "slots": _SLOTS64, "span": (0.02, 0.98), "k_eval": 62,
     "reverse": False, "w": 0.10},
    {"name": "maxspan-v5-reverse", "file": "raptor_ft_coatnet_v5_full_swa.pt",
     "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384,
     "img": 336, "slots": _SLOTS64, "span": (0.02, 0.98), "k_eval": 62,
     "reverse": True, "w": 0.10},
    {"name": "native384-v8", "file": "raptor_ft_coatnet_v8_full_swa.pt",
     "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384,
     "img": 384, "slots": _SLOTS44, "span": (0.06, 0.94), "k_eval": 42,
     "reverse": False, "w": 0.20},
]
_VIEW_W = RUN["raptor_view_w"]
for _arm in ARMS:
    _arm["w"] = float(_VIEW_W[_arm["name"]])

def build_backbone(arch, pretrained=False):
    hybrid = arch.startswith(('maxvit', 'maxxvit', 'coatnet', 'coat_', 'convnext'))
    is_vit = not hybrid and any((k in arch for k in ('vit', 'deit', 'dinov2', 'eva', 'beit')))
    kw = dict(pretrained=pretrained, num_classes=0, in_chans=3)
    if is_vit:
        kw.update(global_pool='token', dynamic_img_size=True)
    else:
        kw.update(global_pool='avg')
    return timm.create_model(arch, **kw)

class RaptorClassifier(nn.Module):

    def __init__(self, backbone, F_dim=768, n=12, drop=0.2):
        super().__init__()
        self.backbone = backbone
        self.norm = nn.LayerNorm(F_dim)
        self.att = nn.Sequential(nn.Linear(F_dim, 256), nn.Tanh(), nn.Dropout(drop), nn.Linear(256, n))
        self.clsW = nn.Parameter(torch.zeros(n, F_dim))
        self.clsb = nn.Parameter(torch.zeros(n))
        nn.init.trunc_normal_(self.clsW, std=0.02)
        self.n = n

    def encode(self, x):
        B, K = x.shape[:2]
        f = self.backbone(x.flatten(0, 1))
        return f.view(B, K, -1)

    def head(self, feats):
        h = self.norm(feats)
        a = self.att(h)
        a = torch.softmax(a, dim=1)
        pooled = torch.einsum('bkn,bkf->bnf', a, h)
        logits = (pooled * self.clsW).sum(-1) + self.clsb
        return logits

    def forward(self, x):
        return self.head(self.encode(x))

def load_model(pt_path, arch_default, res_default, device, ngpu=1):
    ck = torch.load(pt_path, map_location='cpu', weights_only=False)
    arch = ck.get('arch', arch_default)
    if arch != arch_default:
        raise RuntimeError(f'Raptor architecture drift: {arch} != {arch_default}')
    ck_res = int(ck.get('res', res_default))
    bb = build_backbone(arch, pretrained=False)
    model = RaptorClassifier(bb, F_dim=bb.num_features)
    rsna_strict_load(model,ck['model'],str(pt_path))
    rsna_event('raptor_checkpoint',file=str(pt_path),sha256=rsna_sha(pt_path),arch=arch,res=ck_res)
    model.eval().to(device)
    del ck
    gc.collect()
    return (model, ck_res)



def _eval_centers(mask, D, k):
    valid = np.where(mask > 0)[0]
    if len(valid) < 3:
        valid = np.arange(min(3, D))
    lo, hi = (int(valid.min()), int(valid.max()))
    cs = [c for c in range(lo + 1, hi) if c - 1 >= lo and c + 1 <= hi]
    if not cs:
        cs = [max(1, min((lo + hi) // 2, D - 2))]
    idx = np.linspace(0, len(cs) - 1, k).round().astype(int)
    return [cs[i] for i in idx]

def eval_windows(vol, mask, k, res, norm=NORM):
    """Resize each selected source plane once, then gather the unchanged RGB triplets."""
    volume = np.asarray(vol)
    depth = int(volume.shape[0])
    centers = np.asarray(_eval_centers(mask, depth, k), dtype=np.int64)
    centers = np.clip(centers, 1, depth - 2)
    if tuple(volume.shape[-2:]) == (res, res):
        # Native384 needs no interpolation; the existing small per-triplet copy
        # avoids a slower large advanced-index gather and preserves its exact math.
        wins = np.empty((len(centers), 3, res, res), np.float32)
        for j, center in enumerate(centers):
            wins[j] = np.stack([volume[center - 1], volume[center],
                                volume[center + 1]], axis=0).astype(np.float32) / 255.0
        x = torch.from_numpy(wins)
        if norm == 'imagenet':
            x = (x - _MEAN) / _STD
        return x
    triplets = centers[:, None] + np.asarray([-1, 0, 1], dtype=np.int64)
    # Use only planes referenced by the original eval-center recipe.
    unique, inverse = np.unique(triplets.reshape(-1), return_inverse=True)
    source = volume[unique].astype(np.float32) / 255.0
    planes = torch.from_numpy(source)
    if tuple(planes.shape[-2:]) != (res, res):
        resized = torch.empty((len(planes), res, res), dtype=torch.float32)
        for start in range(0, len(planes), 16):
            resized[start:start+16] = F.interpolate(
                planes[start:start+16, None], size=(res, res),
                mode='bilinear', align_corners=False)[:, 0]
        planes = resized
    x = planes[torch.from_numpy(inverse.reshape(-1, 3))]
    if norm == 'imagenet':
        x = (x - _MEAN) / _STD
    return x





def rankpct(x):
    return rsna_rank01(x)

def _make_reader():
    import pydicom, cv2
    from pydicom.pixel_data_handlers.util import apply_modality_lut

    def order_and_meta(sdir):
        fs = glob.glob(sdir + '/*.dcm')
        recs = []
        ps_list = []
        for f in fs:
            try:
                h = pydicom.dcmread(f, stop_before_pixels=True)
                iop = getattr(h, 'ImageOrientationPatient', None)
                ipp = getattr(h, 'ImagePositionPatient', None)
                if iop is not None and ipp is not None and (len(iop) == 6):
                    r = np.array(iop[:3], float)
                    c = np.array(iop[3:], float)
                    n = np.cross(r, c)
                    pos = float(np.dot(np.array(ipp, float), n))
                else:
                    instance = getattr(h, 'InstanceNumber', None)
                    if instance is None:
                        rsna_event('raptor_order_index_fallback', file=str(f)); instance = len(recs)
                    pos = float(instance)
                if not np.isfinite(pos):
                    rsna_event('raptor_order_nonfinite_fallback', file=str(f)); pos = float(len(recs))
                ps = getattr(h, 'PixelSpacing', None)
                ps = float(ps[0]) if ps is not None else 0.5
                ps_list.append(ps)
                recs.append((pos, f, ps))
            except Exception as exc:
                rsna_event('raptor_order_unreadable_skipped', file=str(f), error=f'{type(exc).__name__}: {exc}'); continue
        recs.sort(key=lambda x: (x[0], x[1]))
        med_ps = float(np.median(ps_list)) if ps_list else 0.5
        return ([(f, ps) for _, f, ps in recs], med_ps)

    def read_px(f):
        d = pydicom.dcmread(f)
        a = apply_modality_lut(d.pixel_array, d).astype(np.float32)
        if str(getattr(d, 'PhotometricInterpretation', '')) == 'MONOCHROME1':
            a = a.max() - a
        return a

    def mm_crop_resize(a, ps):
        h, w = a.shape
        cpx = int(round(CROP_MM / max(ps, 0.001)))
        cpx = min(cpx, min(h, w))
        y0 = (h - cpx) // 2
        x0 = (w - cpx) // 2
        a = a[y0:y0 + cpx, x0:x0 + cpx]
        return cv2.resize(a, (IMG, IMG), interpolation=cv2.INTER_AREA)
    return (order_and_meta, read_px, mm_crop_resize)

def _pick_series_for_slot(rows, plane, fluid, used):
    """An unknown protocol flag is not false and must not reach int(NaN)."""
    def flag(value):
        if value is None:
            return None
        if isinstance(value, str) and value.strip().lower() in ('', 'nan', 'none', '<na>', 'unknown'):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            text = str(value).strip().lower()
            if text in ('true', 'yes', 'y', 't'):
                return 1
            if text in ('false', 'no', 'n', 'f'):
                return 0
            return None  # unknown code: neither false nor fatal; the plane-level fallback applies
        if not np.isfinite(number):
            return None
        if number not in (0.0, 1.0):
            return None
        return int(number)
    candidates = [r for r in rows if r['Anatomical_Plane'] == plane
                  and r['SeriesInstanceUID'] not in used]
    if fluid in (0, 1):
        preferred = [r for r in candidates if flag(r.get('Fluid_Sensitive')) == fluid]
        if preferred:
            return preferred[0]
    return candidates[0] if candidates else None



def find_test_root():
    from pathlib import Path
    for value in [os.environ.get('RSNA_COMP_ROOT', ''),
                  '/kaggle/input/competitions/rsna-knee-abnormality-detection',
                  '/kaggle/input/rsna-knee-abnormality-detection']:
        if value and (Path(value)/'test.csv').is_file():
            return value
    raise FileNotFoundError('explicit competition root absent')

def find_weight_file(fname):
    return str(_asset_find_asset(fname))
'''
_KE_NS = {'__name__': '_ke_raptor', 'RUN': RUN, 'coat_w': coat_w, '_asset_find_asset': _asset_find_asset, 'rsna_rank01':rsna_rank01, 'rsna_strict_load':rsna_strict_load, 'rsna_event':rsna_event, 'rsna_sha':rsna_sha}
exec(compile(_KE_SRC, '<raptor>', 'exec'), _KE_NS)
from concurrent.futures import Future as _KeFuture
import threading as _ke_threading

_ke_base_make_reader = _KE_NS['_make_reader']
_ke_order_cache = {}
from collections import OrderedDict as _KeOrderedDict
_ke_pixel_cache = _KeOrderedDict()
_ke_order_cache_lock = _ke_threading.Lock()


def _ke_make_cached_reader():
    """Share immutable DICOM ordering metadata across Raptor views."""
    base_order, read_pixel, crop_resize = _ke_base_make_reader()

    def cached_order(series_dir):
        key = str(series_dir)
        owner = False
        with _ke_order_cache_lock:
            future = _ke_order_cache.get(key)
            if future is None:
                future = _KeFuture()
                _ke_order_cache[key] = future
                owner = True
        if owner:
            try:
                future.set_result(base_order(series_dir))
            except BaseException as error:
                future.set_exception(error)
                with _ke_order_cache_lock:
                    _ke_order_cache.pop(key, None)
                raise
        return future.result()

    def cached_pixel(path):
        global _ke_pixel_cache_bytes
        from collections import OrderedDict
        key=str(path)
        with _ke_order_cache_lock:
            got=_ke_pixel_cache.get(key)
            if got is not None:
                _ke_pixel_cache.move_to_end(key);return got
        value=read_pixel(path)
        if value.ndim!=2 or not _ke_np.isfinite(value).all():
            raise RuntimeError(f'Raptor invalid pixels {path}')
        value.setflags(write=False)
        with _ke_order_cache_lock:
            previous = _ke_pixel_cache.pop(key, None)
            if previous is not None:
                _ke_pixel_cache_bytes -= previous.nbytes
            _ke_pixel_cache[key] = value
            _ke_pixel_cache_bytes += value.nbytes
            while _ke_pixel_cache_bytes > (192 << 20):
                _, removed = _ke_pixel_cache.popitem(last=False)
                _ke_pixel_cache_bytes -= removed.nbytes
        return value
    return cached_order, cached_pixel, crop_resize


_KE_NS['_make_reader'] = _ke_make_cached_reader
if _ke_os.environ.get('RSNA_COMP_ROOT'):
    _KE_NS['find_test_root'] = lambda: _ke_os.environ['RSNA_COMP_ROOT']


def _ke_prepare_windows(arm, study_uid, series, series_root, reader):







    order_and_meta, read_px, _ = reader
    image_size = int(arm['img'])
    slots = list(arm['slots'])
    span_lo, span_hi = map(float, arm['span'])
    used, pools = set(), []
    rows = series.get(study_uid, [])
    for plane, fluid, count in slots:
        record = _KE_NS['_pick_series_for_slot'](rows, plane, fluid, used)
        if record is None:
            pools.append(None)
            continue
        used.add(record['SeriesInstanceUID'])
        files, median_spacing = order_and_meta(
            f"{series_root}/{study_uid}/{record['SeriesInstanceUID']}")
        if not files:
            rsna_event('raptor_acquisition_empty', study=str(study_uid), series=str(record['SeriesInstanceUID'])); pools.append(None); continue
        lo = int(len(files) * span_lo)
        hi = max(int(len(files) * span_hi) - 1, lo)
        original = _ke_np.linspace(lo, hi, count).round().astype(int)
        pools.append((files, median_spacing, lo, hi, original))
    capacities = [0 if p is None else p[3] - p[2] + 1 for p in pools]
    quotas = _dense_allocate(capacities, [s[2] for s in slots], 96)
    volume, sources = [], []
    import cv2
    for pool, quota in zip(pools, quotas):
        if pool is None or not quota:
            continue
        files, median_spacing, lo, hi, original = pool
        picks = lo + _dense_unique_linspace(hi - lo + 1, int(quota))
        arrays = {}
        # Decode union once. Failure is explicit, not an unreported 0.5 score.
        for i in sorted(set(original.tolist()) | set(picks.tolist())):
            arrays[i] = read_px(files[i][0])
        all_pixels = _ke_np.concatenate([arrays[int(i)].ravel() for i in original])
        low, high = _ke_np.percentile(all_pixels, [2.0, 98.0])
        for i in picks:
            path, spacing = files[int(i)]
            a = arrays[int(i)]
            a = _ke_np.clip((a-low)/(high-low+1e-6), 0, 1)
            spacing = spacing if spacing > 0 else median_spacing
            h, w = a.shape
            c = min(int(round(140.0/max(spacing, .001))), min(h, w))
            y, x = (h-c)//2, (w-c)//2
            a = cv2.resize(a[y:y+c, x:x+c], (image_size,image_size),
                           interpolation=cv2.INTER_AREA)
            volume.append((a*255).astype(_ke_np.uint8))
            sources.append(str(path))
    if len(sources) != len(set(sources)):
        raise RuntimeError('capacity-aware sampling selected duplicate source files')
    if len(volume) < 3:
        raise RuntimeError(f'{study_uid}: fewer than three unique source slices')
    volume = _ke_np.stack(volume)
    # Presence means an acquired source, not an intensity test. A genuinely
    # black acquired slice is not interchangeable with padding.
    mask = _ke_np.ones(len(volume), dtype=_ke_np.uint8)
    windows = _KE_NS['eval_windows'](
        volume, mask, k=len(volume)-2, res=int(arm['res']), norm=_KE_NS['NORM'])
    if len(windows) != min(96, sum(capacities))-2:
        raise RuntimeError('capacity-aware sampling Raptor cardinality drift')
    import hashlib, json
    signature = hashlib.sha256(json.dumps(sources, ensure_ascii=False,
                                separators=(',', ':')).encode()).hexdigest()
    audit = {'arm':arm['name'], 'uid':str(study_uid),
             'capacities':capacities, 'quotas':quotas.tolist(),
             'source_count':len(sources), 'source_list_sha256':signature,
             'windows':len(windows), 'duplicate_sources':0}
    identity = (arm['name'], str(study_uid))
    with _ke_input_lock:
        if identity in _ke_input_ids:
            raise RuntimeError(f'Duplicate Raptor preparation: {identity}')
        _ke_input_ids.add(identity)
        with _ke_input_audit_path.open('a') as handle:
            handle.write(json.dumps(audit, sort_keys=True, allow_nan=False) + '\n')
    return windows


def _ke_infer_input(model, xwins, device):
    """Bound backbone workspace; keep all window features for one head call."""
    torch = _KE_NS['torch']
    def _forward(_amp):
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16, enabled=_amp):
            features = [model.backbone(xwins[i:i+8].to(device))
                        for i in range(0,len(xwins),8)]
            feats = torch.cat(features,dim=0).unsqueeze(0)
            return torch.sigmoid(model.head(feats).float())[0].cpu().numpy()
    probabilities = _forward(True)
    if not _ke_np.isfinite(probabilities).all():
        rsna_event('raptor_fp16_nonfinite_retry_fp32')
        probabilities = _forward(False)
    if not _ke_np.isfinite(probabilities).all():
        raise RuntimeError('nonfinite Raptor prediction')
    return probabilities


def _ke_prefetched_windows(arm, test_ids, series, series_root, reader):
    """Bound host memory while decoding one study ahead of its GPU forward."""
    depth = int(_ke_os.environ.get('RSNA_RAPTOR_PREFETCH', '2'))
    if not 1 <= depth <= 4: raise ValueError('Raptor prefetch must be 1..4')
    with _KeThreadPool(max_workers=1) as executor:
        pending = {}
        submit_at = 0
        while submit_at < min(depth, len(test_ids)):
            pending[submit_at] = executor.submit(
                _ke_prepare_windows,
                arm,
                test_ids[submit_at],
                series,
                series_root,
                reader,
            )
            submit_at += 1
        for study_index, study_uid in enumerate(test_ids):
            future = pending.pop(study_index)
            if submit_at < len(test_ids):
                pending[submit_at] = executor.submit(
                    _ke_prepare_windows,
                    arm,
                    test_ids[submit_at],
                    series,
                    series_root,
                    reader,
                )
                submit_at += 1
            yield study_index, study_uid, future


def _ke_run_raptor_arms():
    """Run the four unchanged Raptor arms across exactly two GPUs."""
    torch = _KE_NS['torch']
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError(
            f"optimized Raptor requires exactly two GPUs, got {torch.cuda.device_count()}"
        )
    started = _ke_time.time()
    root = _KE_NS['find_test_root']()
    series_root = root + '/test_series'
    if not _ke_os.path.isdir(series_root):
        series_root = root + '/test_images'
    test = _ke_pd.read_csv(root + '/test.csv')
    test['StudyInstanceUID'] = test['StudyInstanceUID'].astype(str)
    test_ids = test['StudyInstanceUID'].tolist()
    test_series = _ke_pd.read_csv(root + '/test_series.csv')
    test_series['StudyInstanceUID'] = test_series['StudyInstanceUID'].astype(str)
    test_series['SeriesInstanceUID'] = test_series['SeriesInstanceUID'].astype(str)
    series = {
        key: frame.to_dict('records')
        for key, frame in test_series.groupby('StudyInstanceUID')
    }
    sample = root + '/sample_submission.csv'
    columns = ['StudyInstanceUID', *_KE_NS['LAB']]
    if _ke_os.path.exists(sample):
        columns = list(_ke_pd.read_csv(sample, nrows=1).columns)
    arms = list(_KE_NS['ARMS'])
    outputs = [
        _ke_np.full((len(test_ids), len(_KE_NS['LAB'])), np.nan, _ke_np.float32)
        for _ in arms
    ]
    print(
        f"[raptor-fast] {len(test_ids)} studies; balanced arm groups "
        f"cuda:0=[0,2], cuda:1=[1,3]",
        flush=True,
    )

    def run_single(arm_index, device):
        arm = arms[arm_index]
        reader = _KE_NS['_make_reader']()
        weight_path = _KE_NS['find_weight_file'](arm['file'])
        model, resolution = _KE_NS['load_model'](
            weight_path, arm['arch'], arm['res'], device
        )
        if int(resolution) != int(arm['res']):
            raise RuntimeError(
                f"{arm['name']} checkpoint resolution {resolution} != {arm['res']}"
            )
        for study_index, study_uid, future in _ke_prefetched_windows(
            arm, test_ids, series, series_root, reader
        ):
            rsna_deadline('Raptor full-cohort inference')
            try:
                windows = future.result()
                outputs[arm_index][study_index] = _KE_NS['infer_probs'](
                    model, windows, device
                )
                del windows
            except Exception as error:
                rsna_event('raptor_study_failed', arm=str(arm['name']), study=str(study_uid), error=f'{type(error).__name__}: {str(error)[:500]}')
                with _ke_input_lock:
                    _ke_input_ids.add((arm['name'], str(study_uid)))
        del model
        _ke_gc.collect()
        with torch.cuda.device(device):
            torch.cuda.empty_cache()

    def run_shared_maxspan(device):
        first, reverse = arms[0], arms[2]
        comparable = ('file', 'arch', 'res', 'img', 'slots', 'span', 'k_eval')
        if any(first[key] != reverse[key] for key in comparable):
            raise RuntimeError('MaxSpan forward/reverse arms no longer share preprocessing')
        reader = _KE_NS['_make_reader']()
        weight_path = _KE_NS['find_weight_file'](first['file'])
        model, resolution = _KE_NS['load_model'](
            weight_path, first['arch'], first['res'], device
        )
        if int(resolution) != int(first['res']):
            raise RuntimeError(
                f"MaxSpan checkpoint resolution {resolution} != {first['res']}"
            )
        for study_index, study_uid, future in _ke_prefetched_windows(
            first, test_ids, series, series_root, reader
        ):
            rsna_deadline('Raptor full-cohort inference')
            try:
                windows = future.result()
                outputs[0][study_index] = _KE_NS['infer_probs'](
                    model, windows, device
                )
                outputs[2][study_index] = _KE_NS['infer_probs'](
                    model, windows.flip(1).contiguous(), device
                )
                del windows
            except Exception as error:
                rsna_event('raptor_study_failed', arm=str(first['name']), study=str(study_uid), error=f'{type(error).__name__}: {str(error)[:500]}')
                with _ke_input_lock:
                    _ke_input_ids.add((first['name'], str(study_uid))); _ke_input_ids.add((reverse['name'], str(study_uid)))
        del model
        _ke_gc.collect()
        with torch.cuda.device(device):
            torch.cuda.empty_cache()

    def gpu_zero():
        with torch.cuda.device(0):
            run_shared_maxspan(torch.device('cuda:0'))

    def gpu_one():
        with torch.cuda.device(1):
            run_single(1, torch.device('cuda:1'))
            run_single(3, torch.device('cuda:1'))

    with _KeThreadPool(max_workers=2) as executor:
        workers = [executor.submit(gpu_zero), executor.submit(gpu_one)]
        for worker in workers:
            worker.result()

    _ke_np.savez_compressed('/kaggle/working/raptor_raw.npz',
        study_uids=_ke_np.asarray(test_ids), raw_probabilities=_ke_np.stack(outputs))
    for _ai, _arr in enumerate(outputs):
        _bad = ~_ke_np.isfinite(_arr).all(axis=1)
        if _bad.any():
            _fill = _ke_np.nanmean(_ke_np.where(_ke_np.isfinite(_arr), _arr, _ke_np.nan), axis=0) if (~_bad).any() else _ke_np.full(_arr.shape[1], .5, _arr.dtype)
            _arr[_bad] = _ke_np.where(_ke_np.isfinite(_fill), _fill, .5)
            rsna_event('raptor_neutral_fill', arm=str(arms[_ai]['name']), studies=int(_bad.sum()))
    rsna_finite(_ke_np.stack(outputs),'all four Raptor views',probability=True)
    weights = _ke_np.asarray([float(arm['w']) for arm in arms], _ke_np.float64)
    weights /= weights.sum()
    probability_blend = _ke_np.tensordot(
        weights,
        _ke_np.stack([_ke_np.clip(value, 0, 1) for value in outputs]),
        axes=(0, 0),
    )
    ranks = _KE_NS['rankpct'](probability_blend)
    rsna_finite(ranks,'Raptor ranked ensemble',probability=True)
    submission = _ke_pd.DataFrame(ranks.astype(_ke_np.float32), columns=_KE_NS['LAB'])
    submission.insert(0, 'StudyInstanceUID', test_ids)
    submission = submission[columns]
    if submission['StudyInstanceUID'].tolist() != test_ids:
        raise RuntimeError('Raptor study order drift')
    if not _ke_np.isfinite(submission[_KE_NS['LAB']].to_numpy()).all():
        raise RuntimeError('Raptor produced non-finite predictions')
    output = _KePath('/kaggle/working/_raptor.csv')
    submission.to_csv(output, index=False)
    print(
        f"[raptor-fast] wrote {output}; elapsed {_ke_time.time() - started:.1f}s",
        flush=True,
    )


_ke_input_lock = _ke_threading.Lock()
_ke_input_ids = set()
_ke_pixel_cache_bytes = 0
_ke_input_audit_path = _KePath('/kaggle/working/diagnostics/raptor_inputs.jsonl')
_ke_input_audit_path.unlink(missing_ok=True)
_KE_NS['infer_probs'] = _ke_infer_input
for _arm in _KE_NS['ARMS']:
    _arm['parent_k_eval'] = _arm['k_eval']
    _arm['k_eval'] = int(RUN['raptor_k_eval'] or _arm['parent_k_eval'])
rsna_phase('public_raptor', 'START')
_ke_run_raptor_arms()
rsna_phase('public_raptor', 'COMPLETE')
_ke_pixel_cache.clear()
_ke_order_cache.clear()
_ke_gc.collect()
_KePath('/kaggle/working/raptor_input_before_coat.csv').write_bytes(_KePath('/kaggle/working/_raptor.csv').read_bytes())
import json as _asset_json
_KePath('/kaggle/working/raptor_input_summary.json').write_text(
    _asset_json.dumps({'studies':len(_RSNA_TEST_IDS), 'preparations':len(_ke_input_ids),
                    'view_names':[a['name'] for a in _KE_NS['ARMS']],
                    'shared_preparation_views':['maxspan-v5','maxspan-v5-reverse'],
                    'records':str(_ke_input_audit_path)},indent=2))



def _coat_substitute():
    import hashlib as _h, os as _o, subprocess as _sp, sys as _sy
    from pathlib import Path as _P
    import pandas as _pd

    import numpy as _np
    _P('/kaggle/working/_raptor_public.csv').write_bytes(
        _P('/kaggle/working/_raptor.csv').read_bytes())
    _np.savez_compressed('/kaggle/working/btk_v32_rest_input.npz',
        study_uids=_ke_ours.StudyInstanceUID.to_numpy().astype(str),
        labels=_np.asarray(_KE_LAB), values=_ke_ours[_KE_LAB].to_numpy())
    MAN_SHA = '98511a8fdeb9da0e6e70c78d013ff636e1476f31c80b5dc134d294b18c3f284e'
    WHL_SHA = '236c8df54a90f4d02076e6f9c1cc763d794542e886c576a6fee46ec8ff75a7a9'
    raptor = _P('/kaggle/working/_raptor.csv')

    def sha(p):
        d = _h.sha256()
        with _P(p).open('rb') as f:
            for b in iter(lambda: f.read(8 << 20), b''):
                d.update(b)
        return d.hexdigest()

    def find(name, want):
        return _asset_find_asset(name, want)

    man = find('coat_resgated_ep10_top3_manifest.json', MAN_SHA)
    if man is None:
        raise RuntimeError('coat manifest absent or hash mismatch')

    art = man.parent

    whl = find('opencv_python_headless-4.12.0.88-*.whl', WHL_SHA)
    envd = _P('/kaggle/working/_coat_env')
    _sp.run(
        [
            _sy.executable,
            '-m',
            'pip',
            'install',
            '--no-deps',
            '--quiet',
            '--target',
            str(envd),
            str(whl),
        ],
        check=True,
    )

    out = _P('/kaggle/working/_coat_arm.csv')

    import inspect
    runtime_path = _P('/kaggle/working/input_resgated_runtime.py')
    helper_source = 'import numpy as np\n' + '\n'.join(inspect.getsource(f) for f in
        (_dense_allocate, _dense_unique_linspace, _dense_coat_specs))
    runtime_sha = _asset_write_coat_runtime(art,runtime_path,helper_source)

    child = (
        "import sys, json, os\n"
        f"sys.path.insert(0, {str(envd)!r})\n"
        f"sys.path.insert(0, {str(art)!r})\n"
        "import cv2; assert cv2.__version__ == '4.12.0', cv2.__version__\n"
        "import torch; assert torch.cuda.device_count() == 2\n"
        "sys.path.insert(0, '/kaggle/working')\n"
        "import input_resgated_runtime as rt\n"
        "assert rt.base.cv2.__version__ == '4.12.0'\n"
        "from pathlib import Path\n"
        "r = rt.run_submission("
        "competition_root=Path(os.environ['RSNA_COMP_ROOT']) "
        "if os.environ.get('RSNA_COMP_ROOT') "
        "else rt.base.find_competition_root(),\n"
        f"    artifact_root=Path({str(art)!r}), "
        f"output_path=Path({str(out)!r}),\n"
        "    gpu_batch_studies=2, backbone_micro_images=8)\n"
        "assert r['status'] == rt.SUBMISSION_STATUS\n"
        "assert r['models'] == 3\n"
        "assert [i['epoch'] for i in r['checkpoints']] == [4, 6, 8]\n"
        "print('[coat-child] fallback_studies', r['fallback_studies'], flush=True)\n"
        "Path('/kaggle/working/_coat_arm_receipt.json').write_text("
        "json.dumps(r, indent=2))\n"
    )

    env = dict(_o.environ)
    env['CUDNN_CONV_WSCAP_DBG'] = '1024'
    env['RSNA_COMP_ROOT'] = _KE_NS['find_test_root']()
    env['PYTHONPATH'] = f"{envd}:{art}:" + env.get('PYTHONPATH', '')

    import json as _j
    rsna_phase('residual_coat', 'START')
    try:
        _run_required_child([_sy.executable, '-c', child], env,
                            _P('/kaggle/working/resgated_runtime.log'))
        _coat_receipt = _j.loads(_P('/kaggle/working/_coat_arm_receipt.json').read_text())
        if int(_coat_receipt.get('fallback_studies', 0) or 0):
            rsna_event('coat_resgated_fallback_studies', count=int(_coat_receipt['fallback_studies']), failures=list(_coat_receipt.get('failures', []))[:20])
        _coat_resgated_ok = True
    except Exception as _coat_exc:
        rsna_event('coat_resgated_child_failed', error=f'{type(_coat_exc).__name__}: {str(_coat_exc)[:1500]}')
        print(f'[coat-arm] residual CoAt child FAILED; continuing without it (flagged): {type(_coat_exc).__name__}', flush=True)
        _coat_resgated_ok = False
    rsna_phase('residual_coat', 'COMPLETE', ok=_coat_resgated_ok)
    rsna_phase('d4', 'START')
    d4_out = _P('/kaggle/working/d4_input')/'d4.csv'
    try:
        d4_manifest = _asset_find_asset('coatnet_pairfilm_manifest.json',
            '7ada0605bca6b0530569c6454e988ace479606a3328ed591d090e5764fea661d')
        timm_whl = d4_manifest.parent/'timm-1.0.22-py3-none-any.whl'
        if sha(timm_whl) != '888981753e65cbaacfc07494370138b1700a27b1f0af587f4f9b47bc024161d0':
            raise RuntimeError('D4 timm wheel changed')
        d4_env = _P('/kaggle/working/_d4_env')
        _sp.run([_sy.executable,'-m','pip','install','--no-deps','--quiet',
            '--target',str(d4_env),str(timm_whl)],check=True)
        d4_dir = _P('/kaggle/working/d4_input')
        d4_dir.mkdir(exist_ok=True)
        _asset_run_d4(d4_manifest.parent,d4_env,_KE_NS['find_test_root'](),d4_out)
        _coat_d4_ok = True
    except Exception as _d4_exc:
        rsna_event('coat_d4_child_failed', error=f'{type(_d4_exc).__name__}: {str(_d4_exc)[:1500]}')
        print(f'[coat-arm] D4 CoAt child FAILED; continuing without it (flagged): {type(_d4_exc).__name__}', flush=True)
        _coat_d4_ok = False
    rsna_phase('d4', 'COMPLETE', ok=_coat_d4_ok)

    pub = _pd.read_csv(raptor, dtype={'StudyInstanceUID': str})
    pub = rsna_frame(pub,_RSNA_TEST_IDS,_KE_LAB,'public Raptor')
    lab = [c for c in pub.columns if c != 'StudyInstanceUID']
    import numpy as _np
    # Retain BTKD inner blend coefficients; input/new family scores are unvalidated.
    private_alpha = float(RUN['coat_private_alpha'])
    public_rank = pub[lab].rank(method='average', pct=True)
    _members = []
    if _coat_resgated_ok:
        try:
            ours = _pd.read_csv(out, dtype={'StudyInstanceUID': str})
            ours = rsna_frame(ours,_RSNA_TEST_IDS,_KE_LAB,'resgated CoAt')
            if list(ours.columns) != list(pub.columns):
                raise RuntimeError('coat arm column drift')
            ours = ours.set_index('StudyInstanceUID').reindex(pub.StudyInstanceUID.astype(str).tolist()).reset_index()
            if ours[lab].isna().any().any():
                raise RuntimeError('coat arm does not cover every study')
            rsna_save_predictions('coat_resgated_rank',_RSNA_TEST_IDS,ours[lab].to_numpy(),lab)
            _members.append(('resgated_top3', ours[lab].rank(method='average', pct=True)))
        except Exception as _exc:
            rsna_event('coat_resgated_output_rejected', error=f'{type(_exc).__name__}: {str(_exc)[:500]}')
    if _coat_d4_ok:
        try:
            d4 = _pd.read_csv(d4_out,dtype={'StudyInstanceUID':str})
            d4 = rsna_frame(d4,_RSNA_TEST_IDS,_KE_LAB,'D4 CoAt')
            rsna_save_predictions('coat_d4_rank',_RSNA_TEST_IDS,d4[lab].to_numpy(),lab)
            if d4.StudyInstanceUID.duplicated().any() or list(d4.columns)!=list(pub.columns):
                raise RuntimeError('D4 schema/UID drift')
            d4 = d4.set_index('StudyInstanceUID').reindex(pub.StudyInstanceUID.astype(str))
            if d4[lab].isna().any().any():
                raise RuntimeError('D4 missing predictions')
            _members.append(('d4_swa3', d4[lab].reset_index(drop=True).rank(method='average',pct=True)))
        except Exception as _exc:
            rsna_event('coat_d4_output_rejected', error=f'{type(_exc).__name__}: {str(_exc)[:500]}')
    if not _members:
        rsna_event('coat_family_unavailable', note='public Raptor kept as the CoAt/Raptor input (flagged)')
        print('[coat-arm] no CoAt family member available; public Raptor retained (flagged)', flush=True)
        return 0
    if len(_members) == 2:
        private_rank = _asset_half_rank_mix(_members[0][1], _members[1][1])
    else:
        rsna_event('coat_family_partial', member=_members[0][0])
        private_rank = _members[0][1]
    family = pub.copy()
    family[lab] = private_rank
    family.to_csv('/kaggle/working/_coat_family_rank.csv', index=False)
    hybrid = pub.copy()
    hybrid[lab] = (1.0 - private_alpha) * public_rank + private_alpha * private_rank
    if not _np.isfinite(hybrid[lab].to_numpy(_np.float64)).all():
        raise RuntimeError('CoAt/Raptor hybrid contains non-finite values')
    tmp = raptor.with_name('.raptor_coat_hybrid.csv')
    hybrid.to_csv(tmp, index=False)
    _o.replace(tmp, raptor)
    raptor.with_name('_coat_raptor_blend_receipt.json').write_text(_j.dumps({
        'contract': 'public_raptor_private_residual_coat_v32_no_inner_rerank_v1', 'inner_rerank': False, 'private_alpha': private_alpha,
        'within_coat': {m: 0.5 for m, _ in _members} if len(_members) == 2 else {_members[0][0]: 1.0}, 'public_raptor_alpha': 1.0 - private_alpha,
        'study_count': len(pub), 'finding_specific_weights': False, 'members': [m for m, _ in _members]}, indent=2, sort_keys=True) + '\n')
    return len(pub)

_coat_n = _coat_substitute()
print(f'[coat-arm] residual top3 / D4 SWA3 equal rank family; BTKD32 no-inner-rerank; {_coat_n} studies',flush=True)

_ke_theirs = _ke_pd.read_csv('/kaggle/working/_raptor.csv',
                             dtype={'StudyInstanceUID': str})
assert list(_ke_theirs.columns) == list(_ke_ours.columns), 'column drift'
_ke_theirs = _ke_theirs.set_index('StudyInstanceUID').reindex(
    _ke_ours['StudyInstanceUID']).reset_index()
assert _ke_theirs[_KE_LAB].notna().all().all(), 'study identity drift'


_ke_tr = _ke_ours[_KE_LAB].rank(method='average', pct=True)
_ke_cr = _ke_theirs[_KE_LAB].copy()
_blend_transformer = _ke_ours.copy()
_blend_coatnet = _ke_theirs.copy()
_blend_labels = list(_KE_LAB)
_blend_tr = _ke_tr.copy()
_blend_cr = _ke_cr.copy()
_coatnet_weight = {label: coat_w(label) for label in _blend_labels}
_blend_output = _blend_transformer.copy()
for _blend_label in _blend_labels:
    _blend_w = float(_coatnet_weight[_blend_label])
    _blend_output[_blend_label] = (
        (1.0 - _blend_w) * _blend_tr[_blend_label]
        + _blend_w * _blend_cr[_blend_label]
    )
_blend_output[_blend_labels] = _blend_output[_blend_labels].rank(
    method='average', pct=True
)
assert _ke_np.isfinite(
    _blend_output[_blend_labels].to_numpy(_ke_np.float64)
).all()
_blend_output.to_csv(_ke_primary, index=False)