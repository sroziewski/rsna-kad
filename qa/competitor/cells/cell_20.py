SHARED_DINO_PREFIX_LAYERS = 6
SHARED_DINO_PREFIX_SHA256 = '0a55b893bde971c864ea5aeea443f075b17c084d13aa36f30bd1c7863f655cf9'
def _shared_prefix_state_sha256(state):
    digest = hashlib.sha256()
    prefixes = (
        'backbone.embeddings.',
        *(
            f'backbone.encoder.layer.{index}.'
            for index in range(
                SHARED_DINO_PREFIX_LAYERS
            )
        ),
    )
    keys = sorted(
        key
        for key in state
        if key.startswith(prefixes)
    )
    if len(keys) != 113:
        raise WeightsError(
            'shared DINOv2 prefix key-count drift: '
            f'{len(keys)} != 113'
        )
    for key in keys:
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()

def _shared_prefix_eligible(members):
    if not members or not DEVS:
        return False
    return all(
        'state' not in member
        and int(member['config']['unfreeze_last'])
        == SHARED_DINO_PREFIX_LAYERS
        and member['config']['variant'] == 'small'
        and member['config'].get('pool', 'cls_mean')
        in POOL_PARTS
        for member in members
    )

def _shared_dino_prefix(model, images, img_size):
    batch, slots = images.shape[:2]
    values = images.reshape(
        batch * slots,
        *images.shape[2:],
    ).float().div_(255.0)
    if (
        img_size is not None
        and img_size != values.shape[-1]
    ):
        values = F.interpolate(
            values,
            size=(img_size, img_size),
            mode='bilinear',
            align_corners=False,
        )
    values = (
        values - model.mean
    ) / model.std
    hidden = model.backbone.embeddings(values)
    for layer in model.backbone.encoder.layer[
        :SHARED_DINO_PREFIX_LAYERS
    ]:
        hidden = layer(hidden)
        hidden = hidden[0] if isinstance(hidden, (tuple, list)) else hidden
    return hidden

def _shared_dino_tail(
    model,
    hidden,
    batch,
    slots,
    slot_mask,
):
    value = hidden
    for layer in model.backbone.encoder.layer[
        SHARED_DINO_PREFIX_LAYERS:
    ]:
        value = layer(value)
        value = value[0] if isinstance(value, (tuple, list)) else value
    value = model.backbone.layernorm(value)
    patch = value[:, 1:]
    parts = [
        value[:, 0],
        patch.mean(1),
    ]
    if model.pool == 'cls_mean_focal':
        count = max(
            1,
            patch.shape[1] // 8,
        )
        parts.append(
            patch.topk(
                count,
                dim=1,
            ).values.mean(1)
        )
    feature = torch.cat(
        parts,
        dim=1,
    ).reshape(
        batch,
        slots,
        -1,
    )
    return model.head(feature, slot_mask)

@torch.no_grad()
def _predict_shared_dino_members(
    entries,
    cache,
    mask,
    idx,
    dev,
    img_size,
    group,
    starts,
):
    target_idx = {
        target: index
        for index, target in enumerate(TARGETS)
    }
    states = {}
    for member, model, jitter in entries:
        generator = torch.Generator(device=dev)
        jitter_seed = (
            SEED
            + int(
                hashlib.sha256(
                    str(member['id']).encode()
                ).hexdigest()[:8],
                16,
            )
        )
        generator.manual_seed(
            int(jitter_seed) % (2 ** 63 - 1)
        )
        states[member['id']] = {
            'out': [],
            'public': [],
            'soft': [],
            'generator': generator,
            'jitter': jitter,
        }
        model.eval()

    reference = entries[0][1]
    for batch_start in range(
        0,
        len(idx),
        EVAL_BATCH,
    ):
        selected = idx[
            batch_start:
            batch_start + EVAL_BATCH
        ]
        slot_mask = torch.from_numpy(
            mask[selected]
        ).to(dev)
        batch = len(selected)
        slots = cache.shape[1]
        member_windows = {
            member['id']: (
                [],
                [],
                [],
            )
            for member, _, _ in entries
        }

        for start in starts:
            rows = torch.from_numpy(
                np.ascontiguousarray(
                    cache[
                        selected,
                        :,
                        start:start + group,
                    ]
                )
            ).to(dev)
            def _dino_shared_forward(_amp):
                with torch.autocast('cuda', enabled=_amp):
                    _common = _shared_dino_prefix(reference, rows, img_size)
                    return {member['id']: _shared_dino_tail(model, _common, batch, slots, slot_mask).float() for member, model, _ in entries}
            original_logits = _dino_shared_forward(dev.type == 'cuda')
            if dev.type == 'cuda' and not all(bool(torch.isfinite(v).all()) for v in original_logits.values()):
                rsna_event('dino_fp16_nonfinite_retry_fp32', window_start=int(start))
                original_logits = _dino_shared_forward(False)

            for member, model, jitter in entries:
                member_id = member['id']
                view_logits = [
                    original_logits[member_id]
                ]
                if jitter:
                    jittered = augment(
                        rows,
                        generator=states[
                            member_id
                        ]['generator'],
                    )
                    with torch.autocast(
                        'cuda',
                        enabled=dev.type == 'cuda',
                    ):
                        jitter_hidden = (
                            _shared_dino_prefix(
                                reference,
                                jittered,
                                img_size,
                            )
                        )
                        view_logits.append(
                            _shared_dino_tail(
                                model,
                                jitter_hidden,
                                batch,
                                slots,
                                slot_mask,
                            ).float()
                        )
                    if dev.type == 'cuda' and not bool(torch.isfinite(view_logits[-1]).all()):
                        rsna_event('dino_jitter_fp16_nonfinite_dropped', member=str(member_id))
                        view_logits.pop()
                view_probs = [
                    torch.sigmoid(value)
                    for value in view_logits
                ]
                probabilities, logits, originals = (
                    member_windows[member_id]
                )
                logits.append(
                    torch.stack(
                        view_logits
                    ).mean(0)
                )
                probabilities.append(
                    torch.stack(
                        view_probs
                    ).mean(0)
                )
                originals.append(
                    view_probs[0]
                )

        for member, _, _ in entries:
            member_id = member['id']
            win_probs, win_logits, win_originals = (
                member_windows[member_id]
            )
            probabilities = torch.stack(win_probs)
            logits = torch.stack(win_logits)
            originals = torch.stack(win_originals)
            value = (
                torch.sigmoid(logits.mean(0))
                if TTA_POOL == 'logit'
                else probabilities.mean(0)
            )
            value = apply_target_window_pool(
                value,
                probabilities,
                logits,
                originals,
                TTA_TARGET_POOL,
                target_idx,
            )
            states[member_id]['out'].append(
                value.cpu().numpy()
            )
            public_value = apply_target_window_pool(
                originals.mean(0),
                originals,
                logits,
                originals,
                PUBLIC_FRONTIER_TARGET_POOL,
                target_idx,
            )
            states[member_id]['public'].append(
                public_value.cpu().numpy()
            )
            states[member_id]['soft'].append(
                legacy_fold_soft_window_pool(
                    originals,
                    target_idx,
                ).cpu().numpy()
            )

    return {
        member['id']: (
            np.concatenate(states[member['id']]['out']),
            np.concatenate(states[member['id']]['public']),
            np.concatenate(states[member['id']]['soft']),
        )
        for member, _, _ in entries
    }

@torch.inference_mode()
def _check_shared_dino_path(model,dev,img_size,tag):
    model.eval()
    g=torch.Generator().manual_seed(SEED)
    im=torch.randint(0,256,(2,N_SLOT,GROUP,img_size,img_size),generator=g,dtype=torch.uint8).to(dev)
    mask=torch.ones(2,N_SLOT,device=dev);mask[1,-1]=0
    direct=model(im,mask,img_size).float()
    split=_shared_dino_tail(model,_shared_dino_prefix(model,im,img_size),2,N_SLOT,mask).float()
    if not torch.isfinite(split).all() or not torch.allclose(split,direct,rtol=1e-5,atol=2e-5):
        raise RuntimeError(f'{tag}: shared-prefix execution differs from full forward')
    rsna_event('dino_shared_path_parity',member=tag,max_abs=float((split-direct).abs().max()))

def _run_shared_dino_group(
    path,
    members,
    cache,
    mask,
    idx,
    starts,
):
    ordered = sorted(
        members,
        key=lambda member: -(
            member.get('holdout') or 0
        ),
    )
    # run_dinov2 retains only submission_public_0899.csv, whose member
    # predictions are built exclusively from the unaugmented views.
    # Jittering here affected only the intermediate submission.csv that the
    # retained public-frontier file replaces at the end of this stage.
    plans = [
        (
            member,
            False,
        )
        for member in ordered
    ]
    assignments = [
        plans[index::len(DEVS)]
        for index in range(len(DEVS))
    ]
    results = {}
    result_lock = threading.Lock()

    def worker(dev, plans_for_device):
        global _DINOV2_MATCHED_MEMBERS
        loaded = []
        for member, jitter in plans_for_device:
            with BUILD_LOCK:
                checkpoint = torch.load(
                    Path(path) / member['file'],
                    map_location='cpu',
                    weights_only=False,
                )
                state = checkpoint['model']
                prefix_sha256 = (
                    _shared_prefix_state_sha256(
                        state
                    )
                )
                if (
                    prefix_sha256
                    != SHARED_DINO_PREFIX_SHA256
                ):
                    raise WeightsError(
                        f"{member['id']}: frozen prefix "
                        'hash mismatch: '
                        f'{prefix_sha256}'
                    )
                model = build_model(
                    int(
                        member['config'][
                            'unfreeze_last'
                        ]
                    ),
                    variant=member['config'][
                        'variant'
                    ],
                    pool=member['config'].get(
                        'pool',
                        'cls_mean',
                    ),
                    prior=bool(
                        member['config'].get(
                            'prior',
                            False,
                        )
                    ),
                ).to(dev)
                model.load_state_dict(state)
                check_fingerprint(
                    model,
                    dev,
                    IMG,
                    checkpoint.get('fingerprint'),
                    tag=f"{member['id']}: ",
                )
                _check_shared_dino_path(model, dev, IMG, member['id'])
                _DINOV2_MATCHED_MEMBERS += 1
                del checkpoint, state
            loaded.append(
                (member, model, jitter)
            )

        predicted = _predict_shared_dino_members(
            loaded,
            cache,
            mask,
            idx,
            dev,
            IMG,
            GROUP,
            starts,
        )
        with result_lock:
            results.update(predicted)
        del loaded
        gc.collect()
        with torch.cuda.device(dev):
            torch.cuda.empty_cache()

    with ThreadPoolExecutor(max_workers=len(DEVS)) as pool:
        futures=[pool.submit(worker,dev,plan) for dev,plan in zip(DEVS,assignments)]
        for future in futures: future.result()
    if len(results) != len(ordered):
        raise WeightsError(
            'shared DINOv2 worker did not return '
            f'all members: {len(results)} / '
            f'{len(ordered)}'
        )
    log(
        'shared first 6 DINOv2 blocks for '
        f'{len(ordered)} members; original views '
        f'computed once per {len(DEVS)} device(s)'
    )
    return [
        (
            member,
            results[member['id']],
            jitter,
        )
        for member, jitter in plans
    ]

FINGERPRINT_TOL = 0.002
EXPECTED_DINOV2_MEMBERS = 20
_DINOV2_MATCHED_MEMBERS = 0

def fingerprint(model, dev, img_size, n_slot=None, group=None, seed=None):
    n_slot = N_SLOT if n_slot is None else n_slot
    group = GROUP if group is None else group
    seed = SEED if seed is None else seed
    g = torch.Generator().manual_seed(seed)
    imgs = torch.randint(0, 256, (2, n_slot, group, img_size, img_size), generator=g, dtype=torch.uint8).to(dev)
    mask = torch.ones(2, n_slot, device=dev)
    mask[1, -1] = 0.0
    was_training = model.training
    model.eval()
    with torch.no_grad():
        out = model(imgs, mask, img_size).float().cpu().numpy()
    if was_training:
        model.train()
    return out

def check_fingerprint(model, dev, img_size, expected, tol=FINGERPRINT_TOL, tag=''):
    got = fingerprint(model, dev, img_size)
    exp = np.asarray(expected, np.float32)
    if not np.isfinite(got).all() or not np.isfinite(exp).all():
        raise WeightsError(f'{tag}: nonfinite fingerprint')
    if got.shape != exp.shape:
        raise WeightsError(f'{tag}fingerprint shape {got.shape} != stored {exp.shape}: the architecture is not the one these weights were fitted to')
    d = float(np.abs(got - exp).max())
    if d > tol:
        raise WeightsError(f'{tag}fingerprint differs by {d:.4g} (tolerance {tol:g}). The weights load but do not compute what they computed when fitted - preprocessing, resolution or architecture has moved between the two runs.')
    log(f'{tag}fingerprint matches within {d:.2g}')
    return d

class WeightsError(RuntimeError):
    pass

def find_weights(name='manifest.json'):
    import json
    roots = [
        Path('/kaggle/input/datasets/pilkwang/rsna-knee-weights'),
        Path('/kaggle/input/rsna-knee-weights'),
    ]
    valid = []
    for root in roots:
        path = root / name
        if not path.is_file():
            continue
        try:
            man = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            raise WeightsError(f'invalid pinned weights manifest at {path}: {exc}') from exc
        if not isinstance(man.get('members'), list) or len(man['members']) != EXPECTED_DINOV2_MEMBERS:
            raise WeightsError(f'{path} must list exactly {EXPECTED_DINOV2_MEMBERS} members')
        if len({m.get('id') for m in man['members']}) != EXPECTED_DINOV2_MEMBERS:
            raise WeightsError(f'{path} has duplicate or missing member ids')
        missing = [m['file'] for m in man['members'] if not (root / m['file']).is_file()]
        if missing:
            raise WeightsError(
                f'{root} lists {len(man["members"])} members but misses {missing[0]!r}'
            )
        valid.append(root)
    if len(valid) != 1:
        raise WeightsError(f'expected one pinned rsna-knee-weights root, found {valid}')
    return valid[0]
TTA_OVERLAP = True
TTA_POOL = 'prob'
PUBLIC_FRONTIER_TARGET_POOL = {'Fracture': 'max', 'Contusion': 'max', 'Medial Meniscus': 'max', 'Lateral Meniscus': 'max', 'ACL': 'top2', 'MCL': 'top2', "Baker's": 'max'}
TTA_TARGET_POOL = {**PUBLIC_FRONTIER_TARGET_POOL, 'Synovitis': 'original_mean'}
# No-extra-pass diversity branch: smooth focal pooling is evaluated from
# the same no-jitter public-member windows already used by the parent.
LEGACY_FOLD_SOFTPOOL_BETA = {
    'ACL': 6.0, 'MCL': 6.0,
    'Medial Meniscus': 8.0, 'Lateral Meniscus': 8.0,
    "Baker's": 8.0, 'Contusion': 8.0, 'Fracture': 10.0,
}
LEGACY_FOLD_SOFTPOOL_ALPHA = {
    'ACL': 0.20, 'MCL': 0.20,
    'Medial Meniscus': 0.25, 'Lateral Meniscus': 0.25,
    "Baker's": 0.20, 'Contusion': 0.20, 'Fracture': 0.15,
}
LEGACY_MEMBER_WEIGHT_BY_TARGET = {'Lateral Meniscus': 15.0, 'Medial OA': 2.5, 'Lateral OA': 15.0, 'Contusion': 5.0}

def window_starts(n_slice, group, overlap=None):
    overlap = TTA_OVERLAP if overlap is None else overlap
    if overlap and n_slice >= group:
        return list(range(n_slice - group + 1))
    return [g * group for g in range(max(n_slice // group, 1))]

def apply_target_window_pool(values, probs, logits, original_probs, mapping, target_idx):
    for target, mode in mapping.items():
        j = target_idx[target]
        if mode == 'max':
            values[:, j] = probs[:, :, j].max(0).values
        elif mode == 'mean':
            values[:, j] = probs[:, :, j].mean(0)
        elif mode == 'logit_mean':
            values[:, j] = torch.sigmoid(logits[:, :, j].mean(0))
        elif mode == 'original_mean':
            values[:, j] = original_probs[:, :, j].mean(0)
        elif mode in ('top2', 'top3'):
            k = min(int(mode[3:]), probs.shape[0])
            values[:, j] = probs[:, :, j].topk(k, dim=0).values.mean(0)
        else:
            raise ValueError(f'unknown TTA pooling mode for {target}: {mode}')
    return values

def legacy_fold_soft_window_pool(original_probs, target_idx):
    values = original_probs.mean(0).clone()
    for target, beta in LEGACY_FOLD_SOFTPOOL_BETA.items():
        j = target_idx[target]
        x = original_probs[:, :, j]
        weight = torch.softmax(float(beta) * x, dim=0)
        values[:, j] = (weight * x).sum(0)
    return values

@torch.no_grad()
def predict_member(model, cache, mask, idx, dev, img_size, group=None, pool=None, starts=None, jitter=False, jitter_seed=SEED, return_public_frontier=False):
    group = GROUP if group is None else group
    pool = TTA_POOL if pool is None else pool
    starts = window_starts(cache.shape[2], group) if starts is None else list(starts)
    if not starts:
        raise ValueError('predict_member was given no windows to average over')
    target_idx = {t: j for j, t in enumerate(TARGETS)}
    unknown = (set(TTA_TARGET_POOL) | set(PUBLIC_FRONTIER_TARGET_POOL)) - set(target_idx)
    if unknown:
        raise ValueError(f'unknown target(s) in TTA_TARGET_POOL: {unknown}')
    jitter_gen = torch.Generator(device=dev)
    jitter_gen.manual_seed(int(jitter_seed) % (2 ** 63 - 1))
    model.eval()
    out, public_frontier_out, public_soft_out = ([], [], [])
    for b in range(0, len(idx), EVAL_BATCH):
        sel = idx[b:b + EVAL_BATCH]
        m = torch.from_numpy(mask[sel]).to(dev)
        win_probs, win_logits, win_original_probs = ([], [], [])
        for st in starts:
            rows = torch.from_numpy(np.ascontiguousarray(cache[sel, :, st:st + group])).to(dev)
            views = [rows] + ([augment(rows, generator=jitter_gen)] if jitter else [])
            view_probs, view_logits = ([], [])
            for view in views:
                with torch.autocast('cuda', enabled=dev.type == 'cuda'):
                    z = model(view, m, img_size).float()
                if dev.type == 'cuda' and not bool(torch.isfinite(z).all()):
                    rsna_event('dino_legacy_fp16_nonfinite_retry_fp32')
                    with torch.autocast('cuda', enabled=False):
                        z = model(view, m, img_size).float()
                view_logits.append(z)
                view_probs.append(torch.sigmoid(z))
            win_logits.append(torch.stack(view_logits).mean(0))
            win_probs.append(torch.stack(view_probs).mean(0))
            win_original_probs.append(view_probs[0])
        probs = torch.stack(win_probs)
        logits = torch.stack(win_logits)
        original_probs = torch.stack(win_original_probs)
        v = torch.sigmoid(logits.mean(0)) if pool == 'logit' else probs.mean(0)
        v = apply_target_window_pool(v, probs, logits, original_probs, TTA_TARGET_POOL, target_idx)
        out.append(v.cpu().numpy())
        if return_public_frontier:
            public_v = apply_target_window_pool(original_probs.mean(0), original_probs, logits, original_probs, PUBLIC_FRONTIER_TARGET_POOL, target_idx)
            public_frontier_out.append(public_v.cpu().numpy())
            public_soft = legacy_fold_soft_window_pool(original_probs, target_idx)
            public_soft_out.append(public_soft.cpu().numpy())
    primary = np.concatenate(out) if out else np.zeros((0, len(TARGETS)), np.float32)
    if not return_public_frontier:
        return primary
    public_frontier = np.concatenate(public_frontier_out) if public_frontier_out else np.zeros((0, len(TARGETS)), np.float32)
    public_soft = np.concatenate(public_soft_out) if public_soft_out else np.zeros((0, len(TARGETS)), np.float32)
    return (primary, public_frontier, public_soft)
BUILD_LOCK = threading.Lock()
STATE_LOCK = threading.Lock()
LEGACY_BUNDLE_FILE = 'rsna_20260807_v1.pt'
LEGACY_WEIGHT = 0.5

def find_legacy_bundle():
    candidates = [
        Path('/kaggle/input/datasets/pilkwang/rsna-knee-weights') / LEGACY_BUNDLE_FILE,
        Path('/kaggle/input/rsna-knee-weights') / LEGACY_BUNDLE_FILE,
    ]
    hits = [path for path in candidates if path.is_file()]
    if len(hits) > 1:
        raise WeightsError(f'ambiguous legacy bundle: {hits}')
    return hits[0] if hits else None

def legacy_group_members():
    return {}

def _run_member(path, m, dev, Cte, Mte, idx, starts, jitter):
    t0 = time.time()
    with BUILD_LOCK:
        if 'state' in m:
            raise WeightsError(f"{m['id']}: inline legacy state is forbidden")
        ck = torch.load(Path(path) / m['file'], map_location='cpu', weights_only=False)
        state, fp = (ck['model'], ck.get('fingerprint'))
        if fp is None:
            raise WeightsError(f"{m['id']}: stored fingerprint is required")
        model = build_model(int(m['config']['unfreeze_last']), variant=m['config']['variant'], pool=m['config'].get('pool', 'cls_mean'), prior=bool(m['config'].get('prior', False))).to(dev)
        model.load_state_dict(state)
        check_fingerprint(model, dev, IMG, fp, tag=f"{m['id']}: ")
        global _DINOV2_MATCHED_MEMBERS
        _DINOV2_MATCHED_MEMBERS += 1
    t_ready = time.time()
    jitter_seed = SEED + int(hashlib.sha256(str(m['id']).encode()).hexdigest()[:8], 16)
    public_member = 'state' not in m
    predicted = predict_member(model, Cte, Mte, idx, dev, IMG, starts=starts, jitter=jitter, jitter_seed=jitter_seed, return_public_frontier=public_member)
    if public_member:
        p, public_p, public_soft = predicted
    else:
        p, public_p, public_soft = (predicted, None, None)
    t_done = time.time()
    del model, state
    gc.collect()
    if dev.type == 'cuda':
        with torch.cuda.device(dev):
            torch.cuda.empty_cache()
    passes = len(starts) * (2 if jitter else 1)
    return (p, public_p, public_soft, (t_ready - t0, (t_done - t_ready) / max(passes, 1)))

def _combine(per_member):
    all_ids = sorted({s for m in per_member for s in m['ids']})
    pos = {s: i for i, s in enumerate(all_ids)}
    acc = np.zeros((len(all_ids), len(TARGETS)), np.float64)
    tot = np.zeros(len(TARGETS), np.float64)
    for m in per_member:
        target_weight = m.get('target_weight')
        w = np.asarray(target_weight if target_weight is not None else [float(m.get('weight', 1.0))] * len(TARGETS), dtype=np.float64)
        if w.shape != (len(TARGETS),) or np.any(w < 0):
            raise ValueError(f"invalid target weights for {m.get('id')}: {w}")
        rsna_finite(m['pred'],str(m['id'])+' retained raw',probability=True)
        r = pd.DataFrame(m['pred']).rank(pct=True).to_numpy()
        acc[[pos[s] for s in m['ids']]] += r * w[None, :]
        tot += w
    if np.any(tot <= 0):
        raise ValueError(f'at least one target has no ensemble vote: {tot}')
    return (all_ids, acc / tot[None, :])

def combine_public_members_by_fold(per_member, pred_key='pred'):
    # Raw-average the four members within each fold, rank each fold,
    # then give all five folds equal weight.
    all_ids = sorted({study for member in per_member for study in member['ids']})
    position = {study: i for i, study in enumerate(all_ids)}
    groups = {}
    for i, member in enumerate(per_member):
        fold = member.get('fold')
        key = f'fold_{fold}' if fold is not None else f'member_{i}'
        groups.setdefault(key, []).append(member)
    fold_ranks, diagnostics = ([], [])
    for key, members_in_fold in sorted(groups.items()):
        matrices = []
        for member in members_in_fold:
            values = np.full((len(all_ids), len(TARGETS)), np.nan, np.float64)
            values[[position[study] for study in member['ids']]] = np.asarray(member[pred_key], np.float64)
            if np.isnan(values).any():
                raise WeightsError(f"{member.get('id')}: incomplete {pred_key} coverage")
            matrices.append(values)
        raw_fold_mean = np.mean(matrices, axis=0)
        fold_ranks.append(pd.DataFrame(raw_fold_mean).rank(method='average', pct=True).to_numpy(np.float64))
        diagnostics.append({'ensemble_group': key, 'members': len(members_in_fold)})
    if len(fold_ranks) != 5:
        raise WeightsError(f'legacy branch requires five folds, found {len(fold_ranks)}')
    return all_ids, np.mean(fold_ranks, axis=0), pd.DataFrame(diagnostics)

def blend_legacy_frontier_and_soft(frontier_rank, soft_rank):
    output = np.asarray(frontier_rank, np.float64).copy()
    for j, target in enumerate(TARGETS):
        alpha = float(LEGACY_FOLD_SOFTPOOL_ALPHA.get(target, 0.0))
        if alpha:
            output[:, j] = (1.0 - alpha) * frontier_rank[:, j] + alpha * soft_rank[:, j]
    return output

def infer_from_package(path, dev=None):
    man = json.loads((Path(path) / 'manifest.json').read_text())
    members = man['members']
    if len(members) != EXPECTED_DINOV2_MEMBERS:
        raise WeightsError(f'expected {EXPECTED_DINOV2_MEMBERS} DINOv2 members, found {len(members)}')
    log(f'weights package: {len(members)} member(s) from {path}; {len(DEVS)} device(s)')
    test_df = pd.read_csv(ROOT / 'test.csv')
    test_series = pd.read_csv(ROOT / 'test_series.csv')
    plane_map = dict(zip(test_series['SeriesInstanceUID'], test_series['Anatomical_Plane']))
    hte = annotate(walk('test_series'))
    log(f'test header pass: {len(hte)} series')
    groups = {}
    for m in members:
        groups.setdefault(m['pixel_group'], []).append(m)
    groups.update(legacy_group_members())
    per_member, public_frontier_members = ([], [])
    failures = []
    abort = threading.Event()
    est = {'fixed': None, 'win': None}

    def bank(m, ids, pred, starts, jitter, public_pred=None, public_soft=None):
        if not np.isfinite(pred).all():
            rsna_event('dino_member_nonfinite_neutral', member=str(m['id']), rows=int((~np.isfinite(pred)).reshape(len(pred), -1).any(axis=1).sum()))
            pred = np.where(np.isfinite(pred), pred, 0.5)
        if public_pred is not None and not np.isfinite(public_pred).all():
            rsna_event('dino_member_public_nonfinite_neutral', member=str(m['id']))
            public_pred = np.where(np.isfinite(public_pred), public_pred, 0.5)
        rsna_finite(pred, m['id']+' DINO raw', probability=True)
        if public_pred is not None: rsna_finite(public_pred,m['id']+' DINO retained raw',probability=True)
        rsna_save_predictions('dino_'+str(m['id']),ids,public_pred if public_pred is not None else pred,TARGETS)
        if float(np.std(pred)) < 1e-09:
            raise WeightsError(f"{m['id']}: degenerate predictions")
        with STATE_LOCK:
            per_member.append({'id': m['id'], 'fold': m.get('fold'), 'ids': ids, 'pred': pred, 'weight': m.get('weight', 1.0), 'target_weight': m.get('target_weight'), 'holdout': m.get('holdout')})
            if public_pred is not None and len(starts) == len(starts_full):
                if float(np.std(public_pred)) < 1e-09:
                    raise WeightsError(f"{m['id']}: degenerate public-frontier prediction")
                public_frontier_members.append({'id': m['id'], 'fold': m.get('fold'), 'ids': ids, 'pred': public_pred, 'soft_pred': public_soft})
            elif public_pred is not None:
                raise WeightsError(f"{m['id']}: incomplete public-frontier windows")
            log(f"  banked {m['id']} fold {m.get('fold', '?')} ({len(starts)} window(s); {len(per_member)} member(s)")
    for gi, (key, gm) in enumerate(groups.items(), 1):
        cfg = json.loads(key)
        adopt_config_globals(cfg)
        log(f"decode group {gi}/{len(groups)}: {cfg['img']}px x {cfg['slices']} slices, crop {cfg['crop_mm']} mm -> {len(gm)} member(s)")
        st_te, Cte, Mte = build_cache(pick_slots(hte, plane_map), plane_map, lat_of(hte, 'test '), f'test g{gi}')
        idx = np.arange(len(st_te))
        starts_full = window_starts(Cte.shape[2], GROUP)
        if len(groups) == 1 and _shared_prefix_eligible(gm):
            shared_results = _run_shared_dino_group(
                path,
                gm,
                Cte,
                Mte,
                idx,
                starts_full,
            )
            for member, predicted, jitter in shared_results:
                prediction, public, soft = predicted
                bank(
                    member,
                    st_te,
                    prediction,
                    starts_full,
                    jitter,
                    public,
                    soft,
                )
            rsna_release_pixels(Cte)
            del Cte, Mte, shared_results
            gc.collect()
            continue
        pending = sorted(gm, key=lambda m: -(m.get('holdout') or 0))
        left_after = sum((len(g) for j, (_, g) in enumerate(groups.items(), 1) if j > gi))

        def pop_next():
            with STATE_LOCK:
                if not pending:
                    return (None, None, False)
                rsna_deadline('DINO complete-member inference')
                starts, jit = starts_full, False
                return (pending.pop(0), starts, jit)

        def worker(dev):
            while not abort.is_set():
                m, starts, jit = pop_next()
                if m is None:
                    return
                try:
                    p, public_p, public_soft, (fs, ws) = _run_member(path, m, dev, Cte, Mte, idx, starts, jit)
                    with STATE_LOCK:
                        est['fixed'], est['win'] = (fs, ws)
                    bank(m, st_te, p, starts, jit, public_p, public_soft)
                except Exception as exc:
                    with STATE_LOCK:
                        failures.append((m['id'], str(dev), type(exc).__name__, str(exc)))
                    abort.set()
                    return
        threads = [threading.Thread(target=worker, args=(d,)) for d in DEVS]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if failures:
            raise WeightsError(f'DINOv2 member failure; no fallback permitted: {failures[0]}')
        rsna_release_pixels(Cte)
        del Cte, Mte
        gc.collect()
    if _DINOV2_MATCHED_MEMBERS != EXPECTED_DINOV2_MEMBERS:
        raise WeightsError(f'fingerprint gate failed: {_DINOV2_MATCHED_MEMBERS} / {EXPECTED_DINOV2_MEMBERS}')
    if len(public_frontier_members) != EXPECTED_DINOV2_MEMBERS:
        raise WeightsError(f'public-frontier inference incomplete: {len(public_frontier_members)} / {EXPECTED_DINOV2_MEMBERS} members')
    log(f'DINOv2 fail-closed gate PASS: {_DINOV2_MATCHED_MEMBERS}/{EXPECTED_DINOV2_MEMBERS} fingerprints and members')
    frontier_ids, frontier_acc = _combine(public_frontier_members)
    sub = write_submission(frontier_acc, frontier_ids, test_df, '_pipeline_stage.csv')
    log(f'submission.csv = exact no-jitter public-frontier rank mean of {len(public_frontier_members)} member(s); {sub.shape}')
    return sub

def adopt_config_globals(cfg):
    global IMG, CACHE_IMG, GROUP, CACHE_SLICES, N_GROUP, CROP_MM, SLICE_BAND, RULES
    CACHE_IMG = IMG = int(cfg['img'])
    GROUP = int(cfg['group'])
    CACHE_SLICES = int(cfg['slices'])
    N_GROUP = max(CACHE_SLICES // GROUP, 1)
    CROP_MM = float(cfg['crop_mm'])
    SLICE_BAND = tuple((float(x) for x in cfg['band']))
    rules = cfg.get('rules') or RULES_NATIVE
    unknown = {k: v for k, v in rules.items() if k not in RULES_NATIVE or v not in (RULES_NATIVE[k], RULES_LEGACY[k])}
    if unknown:
        raise WeightsError(f'the members record pixel rules this pipeline cannot reproduce: {unknown}')
    RULES = {**RULES_NATIVE, **rules}
    if [s[0] for s in SLOTS] != list(cfg['slots']):
        raise WeightsError(f"the members were fitted on slots {cfg['slots']} and this pipeline defines {[s[0] for s in SLOTS]}; a weight would be read against the wrong slot")