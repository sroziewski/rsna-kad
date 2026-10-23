# ================= DINOSAUR V5: FINE-SPACING RAPTOR RESIDUAL =================
from pathlib import Path as _FSPath
import glob as _fs_glob
import gc as _fs_gc
import json as _fs_json
import os as _fs_os
import numpy as _fs_np
import pandas as _fs_pd
import torch as _fs_torch
import torch.nn as _fs_nn
import torch.nn.functional as _fs_F
import timm as _fs_timm
import pydicom as _fs_pydicom

try:
    import cv2 as _fs_cv2
except Exception:
    _fs_cv2 = None

try:
    from pydicom.pixel_data_handlers.util import apply_modality_lut as _fs_apply_lut
except Exception:
    from pydicom.pixels import apply_modality_lut as _fs_apply_lut

_FS_WORK = _FSPath('/kaggle/working')
_FS_FINAL = _FS_WORK / 'submission.csv'
_FS_ANCHOR = _FS_WORK / 'submission_dinosaur_v5_0942_anchor.csv'
_FS_SAFE = _FS_WORK / 'submission_dinosaur_v5_finespacing_safe.csv'
_FS_MAIN = _FS_WORK / 'submission_dinosaur_v5_finespacing_main.csv'
_FS_STRONG = _FS_WORK / 'submission_dinosaur_v5_finespacing_strong.csv'
_FS_DIAG = _FS_WORK / 'dinosaur_v5_finespacing_diagnostics.json'
_FS_RAW = _FS_WORK / 'dinosaur_v5_finespacing_v9_raw.npz'

_FS_LABELS = [
    'ACL','MCL','Medial Meniscus','Lateral Meniscus',
    'Medial OA','Lateral OA','PF OA','Effusion','Synovitis',
    "Baker's",'Contusion','Fracture'
]

if not _FS_FINAL.is_file():
    raise FileNotFoundError('FineSpacing V5 expected the parent submission.csv first')

_anchor = _fs_pd.read_csv(_FS_FINAL, dtype={'StudyInstanceUID': str})
if _anchor.columns.tolist() != ['StudyInstanceUID', *_FS_LABELS]:
    raise RuntimeError(f'FineSpacing anchor schema drift: {_anchor.columns.tolist()}')
_anchor.to_csv(_FS_ANCHOR, index=False)
_uids = _anchor['StudyInstanceUID'].astype(str).tolist()
if len(_uids) != len(set(_uids)):
    raise RuntimeError('FineSpacing anchor contains duplicate StudyInstanceUID')

def _fs_rank(values):
    values = _fs_np.asarray(values, dtype=_fs_np.float64)
    return _fs_pd.DataFrame(values, columns=_FS_LABELS).rank(
        method='average', pct=True
    ).to_numpy(_fs_np.float64)

def _fs_find_comp_root():
    for p in (
        _FSPath('/kaggle/input/competitions/rsna-knee-abnormality-detection'),
        _FSPath('/kaggle/input/rsna-knee-abnormality-detection'),
    ):
        if (p / 'test.csv').is_file():
            return p
    raise FileNotFoundError('RSNA Knee competition root not found')

def _fs_find_weight():
    direct = [
        _FSPath('/kaggle/input/raptor-knee-finespacing/raptor_ft_coatnet_v9_full.pt'),
        _FSPath('/kaggle/input/raptor-knee-finespacing/1/raptor_ft_coatnet_v9_full.pt'),
        _FSPath('/kaggle/input/datasets/dreaddevelopment/raptor-knee-finespacing/raptor_ft_coatnet_v9_full.pt'),
    ]
    for p in direct:
        if p.is_file():
            return p
    hits = [
        _FSPath(p) for p in _fs_glob.glob(
            '/kaggle/input/**/raptor_ft_coatnet_v9_full.pt', recursive=True
        )
    ]
    return hits[0] if hits else None

_FS_WEIGHT = _fs_find_weight()

if _FS_WEIGHT is None:
    print(
        '[DINOsaur V5] Fine-Spacing checkpoint not mounted; exact 0.942 anchor retained.\n'
        'Attach dataset: dreaddevelopment/raptor-knee-finespacing',
        flush=True,
    )
else:
    _ROOT = _fs_find_comp_root()
    _test = _fs_pd.read_csv(_ROOT / 'test.csv', dtype={'StudyInstanceUID': str})
    _test_ids = _test['StudyInstanceUID'].astype(str).tolist()
    if _test_ids != _uids:
        raise RuntimeError('FineSpacing test UID order does not match 0.942 anchor')

    _tser = _fs_pd.read_csv(
        _ROOT / 'test_series.csv',
        dtype={'StudyInstanceUID': str, 'SeriesInstanceUID': str},
    )
    _series = {
        str(k): v.to_dict('records')
        for k, v in _tser.groupby('StudyInstanceUID')
    }

    _FS_IMG = 336
    _FS_CROP_MM = 140.0
    _FS_SPAN = (0.06, 0.94)
    _FS_SLOTS = [
        ('Sagittal', 1, 22),
        ('Sagittal', 0, 18),
        ('Coronal', 1, 15),
        ('Coronal', 0, 10),
        ('Axial', -1, 15),
    ]
    _FS_MAX = sum(x[2] for x in _FS_SLOTS)  # 80
    _FS_K = 78
    _FS_ARCH = 'coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k'
    _FS_RES = 384
    _FS_MEAN = _fs_torch.tensor([0.485,0.456,0.406]).view(3,1,1)
    _FS_STD = _fs_torch.tensor([0.229,0.224,0.225]).view(3,1,1)

    def _fs_backbone(arch):
        return _fs_timm.create_model(
            arch, pretrained=False, num_classes=0, in_chans=3, global_pool='avg'
        )

    class _FSClassifier(_fs_nn.Module):
        def __init__(self, backbone, n=12, drop=0.2):
            super().__init__()
            d = int(backbone.num_features)
            self.backbone = backbone
            self.norm = _fs_nn.LayerNorm(d)
            self.att = _fs_nn.Sequential(
                _fs_nn.Linear(d,256), _fs_nn.Tanh(), _fs_nn.Dropout(drop),
                _fs_nn.Linear(256,n),
            )
            self.clsW = _fs_nn.Parameter(_fs_torch.zeros(n,d))
            self.clsb = _fs_nn.Parameter(_fs_torch.zeros(n))
            _fs_nn.init.trunc_normal_(self.clsW, std=0.02)

        def forward(self, x):
            B,K = x.shape[:2]
            f = self.backbone(x.flatten(0,1)).view(B,K,-1)
            h = self.norm(f)
            a = _fs_torch.softmax(self.att(h), dim=1)
            pooled = _fs_torch.einsum('bkn,bkf->bnf', a, h)
            return (pooled * self.clsW).sum(-1) + self.clsb

    _ck = _fs_torch.load(_FS_WEIGHT, map_location='cpu', weights_only=False)
    _arch = str(_ck.get('arch', _FS_ARCH))
    _res = int(_ck.get('res', _FS_RES))
    if _arch != _FS_ARCH:
        raise RuntimeError(f'FineSpacing architecture drift: {_arch}')
    if _res != _FS_RES:
        raise RuntimeError(f'FineSpacing resolution drift: {_res}')
    if 'lab' in _ck and list(_ck['lab']) != _FS_LABELS:
        raise RuntimeError(f'FineSpacing label order drift: {_ck["lab"]}')

    _model = _FSClassifier(_fs_backbone(_arch))
    _model.load_state_dict(_ck['model'], strict=True)
    del _ck
    _device = _fs_torch.device('cuda:0' if _fs_torch.cuda.is_available() else 'cpu')
    _model.eval().to(_device)

    def _fs_order(series_dir):
        records = []
        spacings = []
        for fp in _fs_glob.glob(str(series_dir / '*.dcm')):
            try:
                h = _fs_pydicom.dcmread(fp, stop_before_pixels=True)
                iop = getattr(h, 'ImageOrientationPatient', None)
                ipp = getattr(h, 'ImagePositionPatient', None)
                if iop is not None and ipp is not None and len(iop) == 6:
                    r = _fs_np.asarray(iop[:3], float)
                    c = _fs_np.asarray(iop[3:6], float)
                    normal = _fs_np.cross(r, c)
                    pos = float(_fs_np.dot(_fs_np.asarray(ipp, float), normal))
                else:
                    pos = float(getattr(h, 'InstanceNumber', 0) or 0)
                ps = getattr(h, 'PixelSpacing', None)
                ps = float(ps[0]) if ps is not None else 0.5
                records.append((pos, fp, ps))
                spacings.append(ps)
            except Exception:
                records.append((0.0, fp, 0.5))
        records.sort(key=lambda z: z[0])
        med_ps = float(_fs_np.median(spacings)) if spacings else 0.5
        return [(fp, ps) for _,fp,ps in records], med_ps

    def _fs_read(fp):
        ds = _fs_pydicom.dcmread(fp)
        arr = _fs_apply_lut(ds.pixel_array, ds).astype(_fs_np.float32)
        if str(getattr(ds, 'PhotometricInterpretation', '')) == 'MONOCHROME1':
            arr = arr.max() - arr
        return arr

    def _fs_resize(arr, ps):
        h,w = arr.shape
        size = int(round(_FS_CROP_MM / max(float(ps), 1e-3)))
        size = min(size, h, w)
        y0 = (h-size)//2
        x0 = (w-size)//2
        arr = arr[y0:y0+size, x0:x0+size]
        if _fs_cv2 is not None:
            return _fs_cv2.resize(
                arr, (_FS_IMG,_FS_IMG), interpolation=_fs_cv2.INTER_AREA
            )
        t = _fs_torch.from_numpy(arr)[None,None].float()
        return _fs_F.interpolate(
            t, size=(_FS_IMG,_FS_IMG), mode='bilinear', align_corners=False
        )[0,0].numpy()

    def _fs_pick(rows, plane, fluid, used):
        candidates = [
            r for r in rows
            if str(r.get('Anatomical_Plane')) == plane
            and str(r.get('SeriesInstanceUID')) not in used
        ]
        if fluid in (0,1):
            pref = [
                r for r in candidates
                if int(r.get('Fluid_Sensitive',0) or 0) == fluid
            ]
            if pref:
                return pref[0]
        return candidates[0] if candidates else None

    def _fs_study(uid):
        rows = _series.get(uid, [])
        vol = _fs_np.zeros((_FS_MAX,_FS_IMG,_FS_IMG), _fs_np.uint8)
        pos = 0
        used = set()
        for plane,fluid,count in _FS_SLOTS:
            row = _fs_pick(rows, plane, fluid, used)
            if row is None:
                pos += count
                continue
            suid = str(row['SeriesInstanceUID'])
            used.add(suid)
            files, med_ps = _fs_order(_ROOT / 'test_series' / uid / suid)
            if not files:
                pos += count
                continue
            n = len(files)
            lo = int(n * _FS_SPAN[0])
            hi = max(lo, int(n * _FS_SPAN[1]) - 1)
            picks = (
                _fs_np.linspace(lo, hi, count).round().astype(int)
                if n > 1 else _fs_np.zeros(count, dtype=int)
            )
            arrs, pss = [], []
            for p in picks:
                fp,ps = files[min(int(p), n-1)]
                try:
                    arrs.append(_fs_read(fp))
                    pss.append(ps)
                except Exception:
                    arrs.append(None)
                    pss.append(med_ps)
            valid = [a for a in arrs if a is not None]
            if valid:
                allpx = _fs_np.concatenate([a.ravel() for a in valid])
                loq,hiq = _fs_np.percentile(allpx, [2.0,98.0])
            else:
                loq,hiq = 0.0,1.0
            for a,ps in zip(arrs,pss):
                if pos >= _FS_MAX:
                    break
                if a is not None:
                    a = _fs_np.clip((a-loq)/(hiq-loq+1e-6),0,1)
                    a = _fs_resize(a, ps if ps > 0 else med_ps)
                    vol[pos] = (a*255).astype(_fs_np.uint8)
                pos += 1
        mask = (vol.reshape(_FS_MAX,-1).sum(1) > 0).astype(_fs_np.uint8)
        return vol,mask

    def _fs_centers(mask, depth, k):
        valid = _fs_np.where(mask > 0)[0]
        if len(valid) < 3:
            valid = _fs_np.arange(min(3,depth))
        lo,hi = int(valid.min()),int(valid.max())
        centers = [c for c in range(lo+1,hi) if c-1 >= lo and c+1 <= hi]
        if not centers:
            centers = [max(1,min((lo+hi)//2,depth-2))]
        idx = _fs_np.linspace(0,len(centers)-1,k).round().astype(int)
        return [centers[i] for i in idx]

    def _fs_windows(vol,mask):
        centers = _fs_centers(mask,vol.shape[0],_FS_K)
        wins = _fs_np.empty((len(centers),3,_FS_RES,_FS_RES),_fs_np.float32)
        for j,c in enumerate(centers):
            c = max(1,min(int(c),vol.shape[0]-2))
            tri = _fs_np.stack([vol[c-1],vol[c],vol[c+1]],0).astype(_fs_np.float32)/255.
            t = _fs_torch.from_numpy(tri)
            if t.shape[-1] != _FS_RES:
                t = _fs_F.interpolate(
                    t[None], size=(_FS_RES,_FS_RES),
                    mode='bilinear', align_corners=False
                )[0]
            wins[j] = t.numpy()
        x = _fs_torch.from_numpy(wins)
        return (x-_FS_MEAN)/_FS_STD

    @_fs_torch.no_grad()
    def _fs_predict(x):
        x = x.unsqueeze(0).to(_device)
        if _device.type == 'cuda':
            try:
                with _fs_torch.autocast('cuda',dtype=_fs_torch.float16):
                    out = _fs_torch.sigmoid(_model(x).float())[0]
                if _fs_torch.isfinite(out).all():
                    return out.cpu().numpy()
            except RuntimeError:
                _fs_torch.cuda.empty_cache()
        out = _fs_torch.sigmoid(_model(x).float())[0]
        if not _fs_torch.isfinite(out).all():
            raise RuntimeError('FineSpacing non-finite prediction')
        return out.cpu().numpy()

    _fine = _fs_np.full((len(_uids),len(_FS_LABELS)),0.5,_fs_np.float32)
    _fallback = 0
    for i,uid in enumerate(_uids):
        try:
            vol,mask = _fs_study(uid)
            x = _fs_windows(vol,mask)
            _fine[i] = _fs_predict(x)
            del vol,mask,x
        except Exception as exc:
            _fallback += 1
            print(
                f'[FineSpacing] study {i} fallback: {type(exc).__name__}: {exc}',
                flush=True,
            )
        if (i+1) % 100 == 0 or i+1 == len(_uids):
            print(f'[FineSpacing] {i+1}/{len(_uids)}',flush=True)

    del _model
    _fs_gc.collect()
    if _device.type == 'cuda':
        _fs_torch.cuda.empty_cache()

    _fs_np.savez_compressed(
        _FS_RAW,
        study_uids=_fs_np.asarray(_uids),
        labels=_fs_np.asarray(_FS_LABELS),
        probabilities=_fine,
    )

    # Existing public Raptor inside the exact 0.942 graph.
    _old_path = _FS_WORK / '_raptor_public.csv'
    if not _old_path.is_file():
        _old_path = _FS_WORK / '_raptor.csv'
    if not _old_path.is_file():
        raise FileNotFoundError('Existing 0.942 public Raptor prediction was not preserved')

    _old = _fs_pd.read_csv(_old_path,dtype={'StudyInstanceUID':str})
    _old = _old.set_index('StudyInstanceUID').reindex(_uids)
    if _old[_FS_LABELS].isna().any().any():
        raise RuntimeError('Existing Raptor UID coverage drift')

    _old_rank = _fs_rank(_old[_FS_LABELS].to_numpy(_fs_np.float64))
    _fine_rank = _fs_rank(_fine)
    _anchor_rank = _fs_rank(_anchor[_FS_LABELS].to_numpy(_fs_np.float64))

    # Inject *only* the geometry/checkpoint difference from FineSpacing.
    _delta = _fine_rank - _old_rank

    # A small, bounded decorrelation modifier. It changes only the size of an
    # already-small residual; it cannot turn one target into a v9-only route.
    _corr = []
    for j in range(len(_FS_LABELS)):
        x,y = _old_rank[:,j],_fine_rank[:,j]
        c = float(_fs_np.corrcoef(x,y)[0,1]) if _fs_np.std(x)>0 and _fs_np.std(y)>0 else 1.0
        _corr.append(c)
    _corr = _fs_np.asarray(_corr,_fs_np.float64)
    _novelty = _fs_np.clip(1.0-_corr,1e-4,None)
    _med = float(_fs_np.median(_novelty))
    _novelty_scale = _fs_np.clip(_novelty/max(_med,1e-4),0.80,1.20)

    def _fs_candidate(scale,path):
        beta = float(scale) * _novelty_scale[None,:]
        values = _fs_rank(_anchor_rank + beta * _delta)
        out = _anchor.copy()
        out[_FS_LABELS] = values
        if not _fs_np.isfinite(values).all():
            raise RuntimeError('FineSpacing candidate contains non-finite values')
        out.to_csv(path,index=False)
        return out,beta

    _safe,_beta_safe = _fs_candidate(0.06,_FS_SAFE)
    _main,_beta_main = _fs_candidate(0.10,_FS_MAIN)
    _strong,_beta_strong = _fs_candidate(0.16,_FS_STRONG)

    # MAIN is the Kaggle submission. Anchor remains next to it for a clean A/B.
    _main.to_csv(_FS_FINAL,index=False)

    def _corr_map(a,b):
        result = {}
        for j,label in enumerate(_FS_LABELS):
            x,y = a[:,j],b[:,j]
            result[label] = (
                float(_fs_np.corrcoef(x,y)[0,1])
                if _fs_np.std(x)>0 and _fs_np.std(y)>0 else 1.0
            )
        return result

    _diag = {
        'schema':'dinosaur_v5_finespacing_residual_v1',
        'title':'RSNA Knee | DINOsaur V5? 🦖',
        'anchor':'exact 0.942 restructured submission before FineSpacing',
        'checkpoint':str(_FS_WEIGHT),
        'geometry':{
            'slots':[list(x) for x in _FS_SLOTS],
            'max_slices':_FS_MAX,
            'windows':_FS_K,
            'span':list(_FS_SPAN),
            'crop_mm':_FS_CROP_MM,
            'source_px':_FS_IMG,
            'model_px':_FS_RES,
        },
        'fallback_studies':int(_fallback),
        'corr_finespacing_vs_existing_raptor':_corr_map(_fine_rank,_old_rank),
        'corr_finespacing_vs_anchor':_corr_map(_fine_rank,_anchor_rank),
        'novelty_scale':{
            label:float(_novelty_scale[j]) for j,label in enumerate(_FS_LABELS)
        },
        'main_beta':{
            label:float(_beta_main[0,j]) for j,label in enumerate(_FS_LABELS)
        },
        'outputs':{
            'anchor':str(_FS_ANCHOR),
            'safe':str(_FS_SAFE),
            'main':str(_FS_MAIN),
            'strong':str(_FS_STRONG),
            'submission':str(_FS_FINAL),
        },
    }
    _FS_DIAG.write_text(
        _fs_json.dumps(_diag,indent=2,sort_keys=True)+'\n',
        encoding='utf-8',
    )
    print('[DINOsaur V5] Fine-Spacing residual complete',flush=True)
    print('  anchor : submission_dinosaur_v5_0942_anchor.csv',flush=True)
    print('  safe   : submission_dinosaur_v5_finespacing_safe.csv',flush=True)
    print('  main   : submission_dinosaur_v5_finespacing_main.csv <-- submission.csv',flush=True)
    print('  strong : submission_dinosaur_v5_finespacing_strong.csv',flush=True)
    print('  corr(v9,current Raptor):',
          {label:round(float(_corr[j]),4) for j,label in enumerate(_FS_LABELS)},
          flush=True)
