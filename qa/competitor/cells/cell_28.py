from __future__ import annotations
import base64 as _rad_b64
import zlib as _rad_zlib
_RAD_CAL_PAYLOAD = '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'
# BTKD Rad graph: preserve its E13/E11 calibrated feature contract; remove unused twin work.
#
# The pinned reference Rad family is fused with correct-contract E13, then
# the same E13 heads run on the E11 layout at 0.15. No twin/legacy wrapper
# follows it, matching the branch that produced V48's visible submission.

import contextlib as _rad_contextlib
import gc as _rad_gc
import hashlib as _rad_hashlib
import json as _rad_json
import os as _rad_os
import re as _rad_re
import time as _rad_time
from concurrent.futures import ThreadPoolExecutor as _RadThreadPool
from pathlib import Path as _RadPath

import numpy as _rad_np
import pandas as _rad_pd
import pydicom as _rad_pydicom
import torch as _rad_torch
import torch.nn as _rad_nn
import torch.nn.functional as _rad_F
from torchvision.models import resnet50 as _rad_resnet50

_RAD_LABELS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
    'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's",
    'Contusion', 'Fracture',
]
_RAD_ALPHA = RUN['rad_alpha']
_RAD_EXCLUDE = ("Baker's", 'Fracture')
_RAD_HEADS_SHA256 = '54f657826b3458a7ba3d462e198ba380732f2b136246182312704929874a9a2c'
_RAD_REFERENCE_HEADS_SHA256 = '0f465649799ecfbccaac1767844639e7ced44e1bc9babde6e4bac7c5d9b89eaa'
_RAD_ENCODER_SHA256 = '08629f7e7bd3e29b8ee9522ca3f65ce4d010a7ddf74f0ea3c7e3f3d0bbab0734'
_RAD_E13_HEADS_SHA256 = 'ad9f19af73bfdf4e49263c0e45060dc3cb239e1195039b26dc8c0a3a6bcd1a8a'
_RAD_E13_MEMBER_WEIGHT = RUN['rad_e13_member_w']
_RAD_V48_SECOND_ALPHA = RUN['rad_second_alpha']
_RAD_TWIN_ALT_WEIGHT = RUN['rad_twin_alt_w']
_RAD_TOKEN_DIM, _RAD_HEAD_DIM = 2048, 512

_RAD_E11_SLOTS = [
    ('SAG_NOFS', 'Sagittal', None, False),
    ('COR_NOFS', 'Coronal', None, False),
    ('AX_NOFS', 'Axial', None, False),
    ('SAG_FS', 'Sagittal', None, True),
]
_RAD_E11_CROP_MM = 130.0
_RAD_E11_CACHE_SLICES = 8
_RAD_E11_IMG = 224

_RAD_E13_SLOTS = [
    ('SAG_FS', 'Sagittal', None, True),
    ('COR_FS', 'Coronal', None, True),
    ('AX_FS', 'Axial', None, True),
    ('SAG_NOFS', 'Sagittal', None, False),
]
_RAD_E13_CROP_MM = 130.0
_RAD_E13_CACHE_SLICES = 8
_RAD_E13_IMG = 224

# Our independently trained five-fold family.  Its preprocessing and estimator
# are preserved from V35: native DICOM geometry/fat-sat handling and a mean of
# per-fold percentile ranks (rather than v15's rank of the probability mean).
_OUR_N_SLOT, _OUR_N_SLICE, _OUR_IMG = 3, 8, 224

# Exact V40/E10 test representation: three fat-suppressed planes, eight
# acquired slices per plane, full frame, legacy ordering/laterality/fill.
SLOTS = [
    ('SAG_FS', 'Sagittal', None, True),
    ('COR_FS', 'Coronal', None, True),
    ('AX_FS', 'Axial', None, True),
]
N_SLOT = len(SLOTS)
CACHE_SLICES = 8
IMG = CACHE_IMG = 224
CROP_MM = 10_000.0
SLICE_BAND = (0.2, 0.8)
RULES = dict(RULES_LEGACY)
TIME_BUDGET = 8.0 * 3600


def _rad_log(message):
    print(f'[Rad-dual5] {message}', flush=True)


def _rad_sha256(path, chunk=8 << 20):
    digest = _rad_hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(chunk), b''):
            digest.update(block)
    return digest.hexdigest()


def _rad_find_file(name, expected_sha=None, explicit_env=None):
    if explicit_env and _rad_os.environ.get(explicit_env):
        candidates = [_RadPath(_rad_os.environ[explicit_env])]
    else:
        by_name = {
            'ResNet50.pt': [
                '/kaggle/input/datasets/marwanmath/resnet-50-radimagenet-marwan/ResNet50.pt',
                '/kaggle/input/resnet-50-radimagenet-marwan/ResNet50.pt',
            ],
            'v52_radimagenet_heads.pt': [
                '/kaggle/input/datasets/prvsiyan/rsna-knee-v52-radimagenet-heads-20260812/v52_radimagenet_heads.pt',
                '/kaggle/input/rsna-knee-v52-radimagenet-heads-20260812/v52_radimagenet_heads.pt',
                '/kaggle/input/datasets/antoinegg1/rsna-knee-e9-radimagenet-heads-v15/v52_radimagenet_heads.pt',
                '/kaggle/input/rsna-knee-e9-radimagenet-heads-v15/v52_radimagenet_heads.pt',
            ],
            'v52_e11_heads.pt': [
                '/kaggle/input/notebooks/sofiaanjenje/rsna-knee-e13-train/rsna_rad_e11/v52_e11_heads.pt',
                '/kaggle/input/rsna-knee-e13-train/rsna_rad_e11/v52_e11_heads.pt',
                '/kaggle/input/notebooks/sofiaanjenje/rsna-knee-e11-train/rsna_rad_e11/v52_e11_heads.pt',
                '/kaggle/input/rsna-knee-e11-train/rsna_rad_e11/v52_e11_heads.pt',
                '/kaggle/input/datasets/antoinegg1/rsna-knee-e11-diverse-heads-v20/v52_e11_heads.pt',
                '/kaggle/input/rsna-knee-e11-diverse-heads-v20/v52_e11_heads.pt',
            ],
        }
        if name not in by_name:
            raise FileNotFoundError(f'unpinned Rad artifact name: {name}')
        candidates = [_RadPath(path) for path in by_name[name]]
    existing = [path for path in candidates if path.is_file()]
    valid = [
        path for path in existing
        if expected_sha is None or _rad_sha256(path) == expected_sha
    ]
    if len(valid) != 1:
        raise RuntimeError(
            f'expected one verified Rad artifact {name}, found {valid}; existing={existing}'
        )
    return valid[0]


class _RadEncoder(_rad_nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = _rad_nn.Sequential(
            *list(_rad_resnet50(weights=None).children())[:-2]
        )

    def forward(self, image):
        return self.backbone(image).mean(dim=(2, 3))


class _RadHead(_rad_nn.Module):
    def __init__(self):
        super().__init__()
        self.project = _rad_nn.Sequential(
            _rad_nn.LayerNorm(_RAD_TOKEN_DIM),
            _rad_nn.Linear(_RAD_TOKEN_DIM, _RAD_HEAD_DIM),
            _rad_nn.GELU(),
        )
        self.plane = _rad_nn.Parameter(_rad_torch.randn(N_SLOT, _RAD_HEAD_DIM) * .01)
        self.position = _rad_nn.Parameter(_rad_torch.randn(CACHE_SLICES, _RAD_HEAD_DIM) * .01)
        self.query = _rad_nn.Parameter(_rad_torch.randn(len(_RAD_LABELS), _RAD_HEAD_DIM) * .02)
        self.attn = _rad_nn.MultiheadAttention(
            _RAD_HEAD_DIM, 8, dropout=.10, batch_first=True
        )
        self.fuse = _rad_nn.Sequential(
            _rad_nn.LayerNorm(_RAD_HEAD_DIM * 4),
            _rad_nn.Linear(_RAD_HEAD_DIM * 4, _RAD_HEAD_DIM),
            _rad_nn.GELU(),
            _rad_nn.Dropout(.15),
        )
        self.weight = _rad_nn.Parameter(
            _rad_torch.randn(len(_RAD_LABELS), _RAD_HEAD_DIM) * .02
        )
        self.bias = _rad_nn.Parameter(_rad_torch.zeros(len(_RAD_LABELS)))

    def forward(self, feature, mask):
        token = self.project(feature.float())
        token = token.view(len(token), self.plane.shape[0], self.position.shape[0], self.plane.shape[-1])
        token = token + self.plane[None, :, None] + self.position[None, None]
        token = token.flatten(1, 2)
        key_padding = mask <= 0
        all_empty = key_padding.all(1)
        if all_empty.any():
            key_padding = key_padding.clone()
            key_padding[all_empty, 0] = False
        query = self.query.unsqueeze(0).expand(len(token), -1, -1)
        attended = query + self.attn(
            query, token, token, key_padding_mask=key_padding, need_weights=False
        )[0]
        denominator = mask.sum(1, keepdim=True).clamp_min(1).unsqueeze(-1)
        mean = (token * mask.unsqueeze(-1)).sum(1, keepdim=True) / denominator
        mean = mean.expand(-1, len(_RAD_LABELS), -1)
        fused = self.fuse(_rad_torch.cat(
            [attended, mean, _rad_torch.abs(attended - mean), attended * mean], dim=-1
        ))
        return (fused * self.weight.unsqueeze(0)).sum(-1) + self.bias


def _rad_load_public_heads(device, expected_sha):
    heads_path = _rad_find_file('v52_radimagenet_heads.pt', expected_sha)
    payload = _rad_torch.load(heads_path, map_location='cpu', weights_only=True)
    expected = {
        'version': 'v52-radimagenet-resnet50-official-1',
        'targets': _RAD_LABELS,
        'encoder_sha256': _RAD_ENCODER_SHA256,
        'encoder_source_commit': '0ce16f7375db4236e646829d1eca61cdb4282133',
        'img': 224,
        'slices_per_plane': 8,
        'feature': 'global_average_pool',
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f'public-v15 head contract drift for {key}')
    folds = payload.get('folds')
    if not isinstance(folds, list) or len(folds) != 5:
        raise RuntimeError('public-v15 bundle requires exactly five heads')
    if sorted(int(record.get('fold', -1)) for record in folds) != list(range(5)):
        raise RuntimeError('public-v15 fold identity drift')
    heads = []
    for record in folds:
        head = _RadHead().to(device).eval()
        head.load_state_dict(record['state_dict'], strict=True)
        heads.append(head)
    return heads, str(heads_path)


def _rad_load_e13_heads(device):
    # V48 used an unqualified filename shared by E11 and E13. Resolve the
    # intended E13 bundle by content and validate its complete pixel contract.
    heads_path = _rad_find_file('v52_e11_heads.pt', _RAD_E13_HEADS_SHA256)
    payload = _rad_torch.load(heads_path, map_location='cpu', weights_only=False)
    expected = {
        'version': 'e11-radimagenet-resnet50-diverse-1',
        'targets': _RAD_LABELS,
        'encoder_sha256': _RAD_ENCODER_SHA256,
        'slots': [list(slot) for slot in _RAD_E13_SLOTS],
        'crop_mm': _RAD_E13_CROP_MM,
        'img': _RAD_E13_IMG,
        'slices_per_plane': _RAD_E13_CACHE_SLICES,
        'feature': 'global_average_pool',
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f'E13 head contract drift for {key}')
    folds = payload.get('folds')
    if not isinstance(folds, list) or len(folds) != 5:
        raise RuntimeError('E13 bundle requires exactly five heads')
    if sorted(int(record.get('fold', -1)) for record in folds) != list(range(5)):
        raise RuntimeError('E13 fold identity drift')
    heads = []
    for record in folds:
        head = _RadHead().to(device).eval()
        head.load_state_dict(record['state_dict'], strict=True)
        heads.append(head)
    return heads, str(heads_path)


def _rad_load_models(device):
    """Two persistent encoder replicas; unused alternative heads are not loaded."""
    import copy
    path = _rad_find_file('ResNet50.pt', _RAD_ENCODER_SHA256, explicit_env='RSNA_RAD_WEIGHT_PATH')
    encoder = _RadEncoder().eval()
    rsna_strict_load(encoder, _rad_torch.load(path,map_location='cpu',weights_only=True),'Rad encoder')
    if sum(p.numel() for p in encoder.parameters())!=23508032:
        raise RuntimeError('Rad encoder parameter count changed')
    replicas=[encoder.to('cuda:0'), copy.deepcopy(encoder).to('cuda:1')]
    for m in replicas:
        m.requires_grad_(False)
    heads,hpath=_rad_load_public_heads(device,_RAD_REFERENCE_HEADS_SHA256)
    return replicas,heads,str(path),hpath


@_rad_torch.inference_mode()
def _rad_encode(encoder, pixels, slot_mask, device):
    """Preserve parent's <=96 images/device chunks without DataParallel replication."""
    n,slots,slices,height,width=pixels.shape
    token_mask=_rad_np.repeat(slot_mask[:,:,None],slices,axis=2).reshape(n,-1)
    _rad_empty=int((token_mask.sum(1)==0).sum())
    if _rad_empty:
        print(f'[rad] {_rad_empty} study(ies) with no slot evidence encoded as all-zero rows (parent policy)', flush=True)
    valid=_rad_np.flatnonzero(token_mask.reshape(-1)>0)
    flat=pixels.reshape(-1,height,width)
    features=_rad_np.zeros((n,slots*slices,_RAD_TOKEN_DIM),_rad_np.float16)
    def run(indices,d):
        dev=_rad_torch.device(f'cuda:{d}')
        def _rad_forward(_amp):
            with _rad_torch.cuda.device(dev), _rad_torch.inference_mode(), _rad_torch.autocast('cuda',dtype=_rad_torch.float16,enabled=_amp):
                image=_rad_torch.from_numpy(flat[indices]).to(dev).float().div_(127.5).sub_(1.)
                image=image.unsqueeze(1).expand(-1,3,-1,-1).contiguous()
                return encoder[d](image).float().cpu().numpy()
        value=_rad_forward(True)
        half=value.astype(_rad_np.float16)
        if not (_rad_np.isfinite(value).all() and _rad_np.isfinite(half).all()):
            rsna_event('rad_fp16_nonfinite_retry_fp32',tokens=int(len(indices)))
            value=_rad_forward(False); half=_rad_np.clip(value,-65000,65000).astype(_rad_np.float16)
        if not _rad_np.isfinite(half).all():
            rsna_event('rad_nonfinite_zeroed',tokens=int((~_rad_np.isfinite(half).reshape(len(half),-1).all(axis=1)).sum()))
            half=_rad_np.where(_rad_np.isfinite(half),half,0).astype(_rad_np.float16)
        return indices,half
    with _RadThreadPool(max_workers=2) as pool:
        for start in range(0,len(valid),192):
            rsna_deadline('Rad full-cohort inference')
            block=valid[start:start+192]; cut=(len(block)+1)//2
            jobs=[pool.submit(run,ii,d) for d,ii in enumerate((block[:cut],block[cut:])) if len(ii)]
            for job in jobs:
                indices,value=job.result(); features.reshape(-1,_RAD_TOKEN_DIM)[indices]=value
    return features,token_mask.astype(_rad_np.float32)


@_rad_torch.inference_mode()
def _rad_predict_head(head, features, masks, device, batch=64):
    predictions = []
    for start in range(0, len(features), batch):
        image = _rad_torch.from_numpy(features[start:start + batch]).to(device)
        mask = _rad_torch.from_numpy(masks[start:start + batch]).to(device)
        def _head_forward(_amp):
            amp = (_rad_torch.autocast('cuda', enabled=_amp)
                   if device.type == 'cuda' else _rad_contextlib.nullcontext())
            with amp:
                return _rad_torch.sigmoid(head(image if _amp else image.float(), mask)).float().cpu()
        block = _head_forward(True)
        if not bool(_rad_torch.isfinite(block).all()):
            rsna_event('rad_head_fp16_nonfinite_retry_fp32', rows=int(len(image)))
            block = _head_forward(False)
        if not bool(_rad_torch.isfinite(block).all()):
            rsna_event('rad_head_nonfinite_neutral', rows=int((~_rad_torch.isfinite(block)).any(dim=1).sum()))
            block = _rad_torch.where(_rad_torch.isfinite(block), block, _rad_torch.full_like(block, 0.5))
        predictions.append(block)
    return _rad_torch.cat(predictions).numpy()


def _rad_rank_columns(values):
    return rsna_rankpct(values)


def _rad_validate(frame, expected_ids):
    if frame.columns.tolist() != ['StudyInstanceUID', *_RAD_LABELS]:
        raise RuntimeError('V36 submission schema drift')
    ids = frame['StudyInstanceUID'].astype(str).tolist()
    if ids != list(map(str, expected_ids)) or len(ids) != len(set(ids)):
        raise RuntimeError('V36 submission study identity/order drift')
    values = frame[_RAD_LABELS].to_numpy(_rad_np.float64)
    if not _rad_np.isfinite(values).all() or values.min() < 0 or values.max() > 1:
        rsna_event('rad_invalid_values_repaired', nonfinite=int((~_rad_np.isfinite(values)).sum()), out_of_range=int(((values < 0) | (values > 1)).sum()))
        frame[_RAD_LABELS] = _rad_np.clip(_rad_np.where(_rad_np.isfinite(values), values, 0.5), 0, 1)


class _RadLayoutPrefetch:
    """One CPU layout builder overlaps GPU work; GPU head shapes are model-owned."""
    def __init__(self, prepare, configurations, release):
        from concurrent.futures import ThreadPoolExecutor
        self.prepare = prepare
        self.configurations = list(configurations)
        self.release = release
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.pending = None
        self.next_index = 0
        self.acquired = None

    def _submit(self):
        if self.next_index < len(self.configurations):
            config = self.configurations[self.next_index]
            self.next_index += 1
            self.pending = self.executor.submit(self.prepare, *config)

    def __enter__(self):
        self._submit()
        return self

    def take(self):
        if self.acquired is not None:
            raise RuntimeError('release the current Rad layout before taking another')
        if self.pending is None:
            raise StopIteration('all Rad layouts consumed')
        future, self.pending = self.pending, None
        try:
            self.acquired = future.result()
        except BaseException:
            self.executor.shutdown(wait=True, cancel_futures=True)
            raise
        self._submit()
        return self.acquired

    def release_current(self):
        if self.acquired is not None:
            self.release(self.acquired[1])  # (study IDs, pixels, masks)
            self.acquired = None

    def __exit__(self, exc_type, exc, tb):
        self.release_current()
        pending, self.pending = self.pending, None
        self.executor.shutdown(wait=True, cancel_futures=True)
        if pending is not None and not pending.cancelled():
            try:
                extra = pending.result()
                self.release(extra[1])
            except BaseException:
                if exc_type is None:
                    raise
        return False


def _rad_main():
    """BTKD Rad/calibrator graph unchanged; diagnostic-only twin work removed.

    BTKD's E13-on-E11 second input is intentional in its fitted calibrator.
    Keep that pair together; do not claim standalone E13 train/serve parity.
    """
    started=_rad_time.time();work=_RadPath('/kaggle/working');primary=work/'_pipeline_stage.csv'
    test=_rad_pd.read_csv(ROOT/'test.csv',dtype={'StudyInstanceUID':str})
    ids=test.StudyInstanceUID.tolist()
    baseline=rsna_frame(_rad_pd.read_csv(primary,dtype={'StudyInstanceUID':str}),ids,_RAD_LABELS,'Rad parent')
    device=_rad_torch.device('cuda:0')
    encoder,reference_heads,encoder_path,reference_heads_path=_rad_load_models(device)
    tser=_rad_pd.read_csv(ROOT/'test_series.csv',dtype={'StudyInstanceUID':str,'SeriesInstanceUID':str})
    plane=dict(zip(tser.SeriesInstanceUID,tser.Anatomical_Plane))
    # One annotation pass. All three layouts use the identical legacy annotation rules.
    headers=annotate(walk('test_series'))
    side_map=lat_of(headers,'rad common')
    def prepare_layout(layout,crop,tag):
        globals().update(SLOTS=list(layout),N_SLOT=len(layout),CACHE_SLICES=8,IMG=224,CACHE_IMG=224,CROP_MM=float(crop),RULES=dict(RULES_LEGACY))
        value=build_cache(pick_slots(headers,plane),plane,side_map,tag)
        if value[0]!=ids:
            rsna_release_pixels(value[1])
            raise RuntimeError('Rad cache order changed')
        return value
    def encode_layout(prefetch):
        studies,pixels,mask=prefetch.take()
        try:
            f,tm=_rad_encode(encoder,pixels,mask,device)
            return f,tm,int(tm.sum())
        finally:
            prefetch.release_current()
    def predict_heads(heads,f,mask,tag):
        if len(heads)!=5: raise RuntimeError(tag+': five heads required')
        probs=_rad_np.stack([_rad_predict_head(h,f,mask,device) for h in heads])
        rsna_finite(probs,tag,probability=True);rsna_save_predictions(tag,ids,probs,_RAD_LABELS)
        return probs.mean(0)
    first=[('SAG_FS','Sagittal',None,True),('COR_FS','Coronal',None,True),('AX_FS','Axial',None,True)]
    # E13 has four slots; construct its model before the producer mutates globals.
    globals().update(SLOTS=list(_RAD_E13_SLOTS),N_SLOT=4,CACHE_SLICES=8)
    e13_heads,e13_path=_rad_load_e13_heads(device)
    configurations=[(first,10000.,'rad_e10'),(_RAD_E13_SLOTS,130.,'rad_e13'),(_RAD_E11_SLOTS,130.,'rad_legacy_second')]
    with _RadLayoutPrefetch(prepare_layout,configurations,rsna_release_pixels) as prefetch:
        f,m,tokens0=encode_layout(prefetch)
        reference_rank=_rad_rank_columns(predict_heads(reference_heads,f,m,'rad_e10_raw'))
        del f,m,reference_heads
        f,m,tokens1=encode_layout(prefetch)
        e13_rank=_rad_rank_columns(predict_heads(e13_heads,f,m,'rad_e13_raw'))
        reference_rank=_rad_rank_columns((1-_RAD_E13_MEMBER_WEIGHT)*reference_rank+_RAD_E13_MEMBER_WEIGHT*e13_rank)
        del f,m,e13_rank
        baseline_rank=_rad_rank_columns(baseline[_RAD_LABELS].to_numpy())
        candidate=baseline.copy()
        for j,label in enumerate(_RAD_LABELS):
            if label not in _RAD_EXCLUDE:
                candidate[label]=(1-_RAD_ALPHA)*baseline_rank[:,j]+_RAD_ALPHA*reference_rank[:,j]
        _rad_validate(candidate,ids)
        f,m,tokens2=encode_layout(prefetch)
        second_rank=_rad_rank_columns(predict_heads(e13_heads,f,m,'rad_legacy_second_raw'))
        del f,m,e13_heads
    branch=candidate.copy()
    branch[_RAD_LABELS]=_rad_rank_columns((1-_RAD_V48_SECOND_ALPHA)*_rad_rank_columns(candidate[_RAD_LABELS].to_numpy())+_RAD_V48_SECOND_ALPHA*second_rank)
    cal=_rad_json.loads(_rad_zlib.decompress(_rad_b64.b64decode(_RAD_CAL_PAYLOAD)).decode())
    protocol=_rad_pd.DataFrame(index=_rad_pd.Index(ids,name='StudyInstanceUID'))
    protocol['n_series']=tser.groupby('StudyInstanceUID').size().reindex(ids).fillna(0)
    for pl in ('Sagittal','Coronal','Axial'):
        part=tser[tser.Anatomical_Plane.astype(str)==pl]
        protocol[f'n_{pl[:3]}']=part.groupby('StudyInstanceUID').size().reindex(ids).fillna(0)
    for flag in ('Fat_Suppression','Fluid_Sensitive'):
        marked=tser[_rad_pd.to_numeric(tser[flag],errors='coerce').fillna(0)>0]
        protocol[flag[:3]]=marked.groupby('StudyInstanceUID').size().reindex(ids).fillna(0)
        for pl in ('Sagittal','Coronal','Axial'):
            part=marked[marked.Anatomical_Plane.astype(str)==pl]
            protocol[f'{flag[:3]}_{pl[:3]}']=part.groupby('StudyInstanceUID').size().reindex(ids).fillna(0)
    if list(protocol.columns)!=list(cal['protocol_columns']): raise RuntimeError('calibrator protocol drift')
    mean=(baseline_rank+reference_rank+second_rank)/3.
    blocks=[baseline_rank,reference_rank,second_rank,reference_rank-baseline_rank,second_rank-baseline_rank,mean]
    for group in cal['groups']:
        blocks.append(mean[:,[_RAD_LABELS.index(t) for t in group]].mean(axis=1,keepdims=True))
    blocks.append(protocol.to_numpy(_rad_np.float64))
    x=_rad_np.concatenate(blocks,axis=1); center=_rad_np.asarray(cal['mean']);spread=_rad_np.asarray(cal['scale']);coef=_rad_np.asarray(cal['coef']);bias=_rad_np.asarray(cal['intercept'])
    if x.shape[1]!=coef.shape[1] or not bool((spread>0).all()): raise RuntimeError('calibrator shape/scale drift')
    adjusted=_rad_rank_columns(((x-center)/spread)@coef.T+bias)
    values=branch[_RAD_LABELS].to_numpy(_rad_np.float64).copy()
    for j,label in enumerate(_RAD_LABELS):
        if label in set(cal['gate']):
            _cw = float(RUN['rad_cal_w'])
            values[:,j]=(1.0-_cw)*values[:,j]+_cw*adjusted[:,j]
    final=branch.copy();final[_RAD_LABELS]=_rad_rank_columns(values)
    _rad_validate(final,ids);rsna_save_predictions('btkd_parent_after_rad',ids,final[_RAD_LABELS].to_numpy(),_RAD_LABELS)
    tmp=primary.with_suffix('.tmp');final.to_csv(tmp,index=False);_rad_os.replace(tmp,primary)
    globals()['V18_CALIBRATOR_APPLIED']=True
    rsna_json(work/'v558_rad_receipt.json',{'status':'COMPLETE','graph':'BTKD legacy Rad plus its matched calibrator preserved','reference_heads_sha256':_RAD_REFERENCE_HEADS_SHA256,'e13_sha256':_RAD_E13_HEADS_SHA256,'encoder_sha256':_RAD_ENCODER_SHA256,'legacy_cross_layout_pass_preserved':True,'standalone_cross_layout_train_parity_claimed':False,'unused_alternate_family_removed':True,'encoder_replicas':2,'layout_cpu_prefetch':True,'model_owned_head_shape':True,'tokens':[tokens0,tokens1,tokens2],'seconds':_rad_time.time()-started})
    del encoder;_rad_gc.collect()
    for d in range(2):
        with _rad_torch.cuda.device(d): _rad_torch.cuda.empty_cache()
    _rad_log('BTKD reference Rad/calibrator complete; unused alternate-head branch removed')


rsna_phase('radimagenet', 'START')
_rad_main()
rsna_phase('radimagenet', 'COMPLETE')