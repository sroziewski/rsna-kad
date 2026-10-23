AMP_PREF = 'bf16'  # BTKD precision retained, including T4 execution; no unvalidated AMP change.

def amp_for(dev):
    if not str(dev).startswith('cuda'):
        return (torch.float32, False)
    cc = torch.cuda.get_device_capability(dev)
    if AMP_PREF == 'bf16':
        return (torch.bfloat16, True)
    if AMP_PREF == 'fp16':
        return (torch.float16, True)
    if AMP_PREF == 'fp32':
        return (torch.float32, False)
    return (torch.bfloat16 if cc >= (8, 0) else torch.float16, True)
AMP_DT, AMP_ON = amp_for(DEV)
WORKERS = max(1, min(4, os.cpu_count() or 4))
CHUNK = 48
MICRO = 8
import copy as _speed_a5_copy
from concurrent.futures import ThreadPoolExecutor as _SpeedA5Pool
if torch.cuda.device_count() != 2:
    raise RuntimeError('A5 optimized path requires exactly two GPUs')
_speed_a5_second = [_speed_a5_copy.deepcopy(m).to('cuda:1').eval() for m in models]
_speed_a5_models = [[m.to('cuda:0').eval() for m in models], _speed_a5_second]
del _speed_a5_second
models = _speed_a5_models[0]
_speed_a5_pool = _SpeedA5Pool(max_workers=2)
print(f"device {DEV} | amp {str(AMP_DT).split('.')[-1]} (on={AMP_ON}) | workers {WORKERS} | chunk {CHUNK} | micro {MICRO}")

def _norm_(im):
    k = CFG.get('norm', 'none')
    if k == 'zscore':
        m = (im > 0).float()
        n = m.sum(dim=(1, 2, 3), keepdim=True).clamp(min=1.0)
        mu = (im * m).sum(dim=(1, 2, 3), keepdim=True) / n
        var = (((im - mu) * m) ** 2).sum(dim=(1, 2, 3), keepdim=True) / n
        return (im - mu) / (var.sqrt() + 1e-06) * m
    if k == 'imagenet':
        m = (im > 0).float()
        return (im - 0.485) / 0.229 * m
    return im

@torch.no_grad()
def _speed_a5_micro(images, masks, *, dev, model_group, amp_dtype):
    """Predict acquired rows only; NaN is a sentinel exclusively for absent inputs."""
    active = np.flatnonzero((masks > 0).any(axis=1))
    out = np.full((len(model_group), len(masks), len(LABELS)), np.nan, np.float32)
    if not len(active):
        return out
    ims, slots, sidx, vms = [], [], [], []
    for compact_index, original_index in enumerate(active):
        present = np.flatnonzero(masks[original_index] > 0)
        blk = images[original_index][present]
        ims.append(torch.from_numpy(blk))
        vms.append(torch.from_numpy(blk.reshape(blk.shape[0], blk.shape[1], -1).max(2) > 0))
        slots.append(torch.from_numpy(present + 1).long())
        sidx.append(torch.full((len(present),), compact_index, dtype=torch.long))
    im = _norm_(torch.cat(ims).to(dev, non_blocking=True).float().div_(255.0))
    sl = torch.cat(slots).to(dev)
    si = torch.cat(sidx).to(dev)
    vm = torch.cat(vms).to(dev)
    sm = torch.zeros(len(sl), CFG.get('n_meta', 0), device=dev)
    per = torch.zeros(len(model_group), len(active), len(LABELS), device=dev, dtype=torch.float32)
    def _a5_forward(enabled):
        with torch.autocast('cuda' if str(dev).startswith('cuda') else 'cpu',
                            dtype=amp_dtype, enabled=enabled):
            for fold_index, model in enumerate(model_group):
                per[fold_index] = torch.sigmoid(model(im, sl, sm, si, len(active), vm=vm).float())
        return per.cpu().numpy()
    got = _a5_forward(AMP_ON)
    if AMP_ON and not np.isfinite(got).all():
        rsna_event('a5_fp16_nonfinite_retry_fp32', rows=int(len(active)))
        got = _a5_forward(False)
    _bad = ~np.isfinite(got)
    if _bad.any():
        rsna_event('a5_nonfinite_neutral', rows=int(_bad.any(axis=(0, 2)).sum()))
        got = np.where(_bad, 0.5, got)
    rsna_finite(got, 'A5 acquired-row probabilities', probability=True)
    out[:, active] = got
    return out

"""Two GPU owners, preserving every original eight-study microbatch."""


def speed_a5_ranges(n, micro=8, devices=2):
    if n < 0 or micro < 1 or devices < 1:
        raise ValueError((n, micro, devices))
    return [[(start, min(start + micro, n))
             for i, start in enumerate(range(0, n, micro)) if i % devices == d]
            for d in range(devices)]


def speed_a5_predict(images, masks, models_by_device, executor, dtype=None):
    """No shard-local ranks; output remains [fold, original study, finding]."""
    count = len(masks)
    assert len(images) == count
    output = np.full((len(models_by_device[0]), count, len(LABELS)), np.nan, np.float32)
    plans = speed_a5_ranges(count, MICRO, len(models_by_device))

    def owner(device_index, ranges):
        device = torch.device(f"cuda:{device_index}")
        result = []
        with torch.cuda.device(device):
            for start, stop in ranges:
                value = _speed_a5_micro(images[start:stop], masks[start:stop],
                    dev=device, model_group=models_by_device[device_index],
                    amp_dtype=AMP_DT if dtype is None else dtype)
                result.append((start, stop, value))
        return result

    tasks = [executor.submit(owner, d, ranges) for d, ranges in enumerate(plans) if ranges]
    for future in tasks:
        for start, stop, value in future.result():
            output[:, start:stop] = value
    return output

def predict(images, masks):
    return speed_a5_predict(images, masks, _speed_a5_models, _speed_a5_pool)

def _a5_validate_applicability(raw, eligible):
    """Never confuse an absent input with failed inference on an acquired input."""
    x = np.asarray(raw)
    m = np.asarray(eligible, dtype=bool)
    if x.ndim != 3 or x.shape[1] != len(m):
        raise ValueError('A5 eligibility/raw shape mismatch')
    if m.any():
        rsna_finite(x[:, m], 'A5 acquired-row complete tensor', probability=True)
    if (~m).any() and not np.isnan(x[:, ~m]).all():
        raise RuntimeError('A5 absent rows must retain explicit NaN sentinels')
    return m

def _a5_rank_available(raw, eligible):
    eligible = _a5_validate_applicability(raw, eligible)
    result = np.full((len(eligible), raw.shape[-1]), .5, dtype=np.float64)
    if eligible.any():
        result[eligible] = 0.0
        for fold in raw:
            result[eligible] += rsna_rank01(fold[eligible])
        result[eligible] /= raw.shape[0]
    return result

def _a5_blend_available(parent, a5_values, eligible, weight):
    """Parent-only on metadata-proven absence; no silent fallback on model error."""
    base = np.asarray(parent, dtype=np.float64)
    a5 = np.asarray(a5_values, dtype=np.float64)
    mask = np.asarray(eligible, dtype=bool)
    if base.shape != a5.shape or mask.shape != (len(base),):
        raise ValueError('A5 merge shape mismatch')
    rsna_finite(base, 'A5 parent probabilities', probability=True)
    if not mask.any() or weight == 0:
        return base.copy()
    rsna_finite(a5[mask], 'A5 applicable ranks', probability=True)
    base_rank = pd.DataFrame(base).rank(method='average', pct=True).to_numpy()
    a5_rank = pd.DataFrame(a5[mask]).rank(method='average', pct=True).to_numpy()
    result = base_rank.copy()
    result[mask] = (1-weight)*base_rank[mask] + weight*a5_rank
    rsna_finite(result, 'A5 applicability-aware merge', probability=True)
    return result

# Macro ROC-AUC depends on ordering, so combine fold orderings rather
# than allowing a fold's probability scale to dominate the mean.
preds = np.full((len(models), len(studies), len(LABELS)), np.nan, np.float32)
_a5_eligible = np.zeros(len(studies), dtype=bool)
t0, done = (time.time(), 0)
def submit_study_block(executor, start):
    """Keep one bounded CPU decode block ahead of GPU inference."""
    block = studies[start:start + CHUNK]
    futures = [
        executor.submit(
            build_study,
            (index, study, by.get(study, [])),
        )
        for index, study in enumerate(block)
    ]
    return start, block, futures

# CPU decode threads avoid forking a process after CUDA/model initialization.
with A5DecodePool(max_workers=WORKERS) as ex:
    pending_block = submit_study_block(ex, 0)
    while pending_block is not None:
        rsna_deadline('A5 full-cohort inference')
        c0, block, futs = pending_block
        next_start = c0 + len(block)
        pending_block = (
            submit_study_block(ex, next_start)
            if next_start < len(studies)
            else None
        )
        imgs = np.zeros((len(block), N_SLOT, N_SLICE, SIZE, SIZE), np.uint8)
        msks = np.zeros((len(block), N_SLOT), np.uint8)
        _fut_index = {f: i for i, f in enumerate(futs)}
        for f in as_completed(futs):
            try:
                i, a, k = f.result()
                imgs[i], msks[i] = (a, k)
            except Exception as e:
                i = _fut_index[f]
                rsna_event('a5_study_failed', study=str(block[i]), error=f'{type(e).__name__}: {str(e)[:500]}')
                print(f'[a5] {block[i]}: build failed, treated as A5-absent (flagged): {type(e).__name__}: {str(e)[:200]}', flush=True)
                imgs[i] = 0; msks[i] = 0
        _a5_eligible[c0:c0 + len(block)] = (msks > 0).any(axis=1)
        preds[:, c0:c0 + len(block)] = predict(imgs, msks)
        _a5_validate_applicability(preds[:, c0:c0 + len(block)], _a5_eligible[c0:c0 + len(block)])
        done += len(block)
        el = time.time() - t0
        print(f'  {done:,}/{len(studies):,}  {el / 60:.1f}m  eta {el / done * (len(studies) - done) / 60:.1f}m', flush=True)
        del imgs, msks
        gc.collect()
print(f'\ninference done in {(time.time() - t0) / 60:.1f} min')
np.savez_compressed('/kaggle/working/speed_a5_raw.npz',
    study_uids=np.asarray(studies), raw_probabilities=preds, applicable=_a5_eligible)
_speed_a5_pool.shutdown(wait=True)
del _speed_a5_models, models, m, enc, z
gc.collect()
for _speed_device in range(2):
    with torch.cuda.device(_speed_device):
        torch.cuda.empty_cache()
A5_W = RUN['a5_w']
A5_LABELS = list(LABELS)
_a5_ok = _a5_validate_applicability(preds, _a5_eligible)
_a5_rank_mean = _a5_rank_available(preds, _a5_eligible)
A5_PREDS = dict(zip(sub_df['StudyInstanceUID'].astype(str), _a5_rank_mean.astype(np.float32)))
A5_AVAILABLE = dict(zip(sub_df['StudyInstanceUID'].astype(str), _a5_eligible.tolist()))
rsna_save_predictions('a5_rank_mean', studies, _a5_rank_mean, LABELS)
rsna_json('/kaggle/working/diagnostics/a5_applicability.json', {
    'unavailable_uids':[u for u, ok in A5_AVAILABLE.items() if not ok],
    'available_count':int(_a5_eligible.sum()), 'model_failure_fallback_allowed':False,
    'rank_placeholder_for_unavailable':0.5, 'placeholder_used_in_blend':False})
rsna_phase('a5', 'COMPLETE', applicable=int(_a5_eligible.sum()), absent=int((~_a5_eligible).sum()))
for _a5k, _a5v in _A5_SAVED.items():
    globals()[_a5k] = _a5v
del _A5_SAVED, _a5k, _a5v