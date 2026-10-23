import numpy as np

def _dense_allocate(capacities, proportions, target):
    capacities = np.asarray(capacities, dtype=np.int64)
    proportions = np.asarray(proportions, dtype=np.float64)
    if capacities.ndim != 1 or capacities.shape != proportions.shape or np.any(capacities < 0) or (not np.isfinite(proportions).all()) or np.any(proportions <= 0) or (int(target) < 1):
        raise ValueError('invalid capacity-aware sampling allocation')
    ideal = int(target) * proportions / proportions.sum()
    initial = np.floor(ideal).astype(np.int64)
    for i in np.argsort(-(ideal - initial), kind='stable')[:int(target) - int(initial.sum())]:
        initial[i] += 1
    quotas = np.minimum(initial, capacities)
    wanted = min(int(target), int(capacities.sum()))
    deficit = np.zeros(len(quotas), dtype=np.float64)
    while int(quotas.sum()) < wanted:
        donors = quotas < capacities
        weights = np.where(donors, proportions, 0.0)
        deficit[donors] += weights[donors] / weights.sum()
        chosen = int(np.argmax(np.where(donors, deficit, -np.inf)))
        quotas[chosen] += 1
        deficit[chosen] -= 1
    assert int(quotas.sum()) == wanted and np.all(quotas <= capacities)
    return quotas

def _dense_unique_linspace(capacity, count):
    if not 0 <= count <= capacity:
        raise ValueError('cannot create unique picks beyond physical capacity')
    result = np.linspace(0, capacity - 1, count).round().astype(np.int64)
    if count and (result.min() < 0 or result.max() >= capacity):
        raise AssertionError('out-of-range source')
    assert len(np.unique(result)) == count
    return result

def _dense_coat_specs(study):
    import raptor_light224 as sampler
    capacities = []
    for row in study.series_rows:
        if int(row) < 0:
            capacities.append(0)
        else:
            centers, _, _, _ = sampler._candidate_centers(int(row), study.offsets, study.valid, study.source_depth)
            capacities.append(len(centers))
    if sum(capacities) < 1:
        raise ValueError('no usable CoAt triplets')
    quotas = _dense_allocate(capacities, (20, 12, 12, 8, 16, 8), 94)
    specs = sampler.sample_eval_windows(study.series_rows, study.offsets, study.valid, study.source_depth, count=None, slot_budgets=np.maximum(quotas, 1).tolist())
    if len(specs) != min(94, sum(capacities)):
        raise RuntimeError('capacity-aware sampling window-count drift')
    keys = [(s.series_row, s.center_local) for s in specs]
    if len(keys) != len(set(keys)):
        raise RuntimeError('capacity-aware sampling repeated a native center')
    for s in specs:
        start = int(study.offsets[s.series_row])
        if not np.all(study.valid[start + np.asarray(s.local_indices)]):
            raise RuntimeError('capacity-aware sampling invalid source channel')
    return specs
INFERRED96_WIDTHS = (27, 21, 18, 12, 18)

def _dense_quotas(capacities) -> np.ndarray:
    capacities = np.asarray(capacities, dtype=np.int64)
    base = np.asarray(INFERRED96_WIDTHS, dtype=np.int64)
    if capacities.shape != (5,) or bool((capacities < 0).any()):
        raise ValueError('expected five nonnegative source capacities')
    quotas = np.minimum(base, capacities)
    target = min(96, int(capacities.sum()))
    deficit = np.zeros(5, dtype=np.float64)
    while int(quotas.sum()) < target:
        donors = quotas < capacities
        weights = np.where(donors, base, 0).astype(np.float64)
        deficit[donors] += weights[donors] / weights.sum()
        selected = int(np.argmax(np.where(donors, deficit, -np.inf)))
        quotas[selected] += 1
        deficit[selected] -= 1
    if int(quotas.sum()) != target or bool((quotas > capacities).any()):
        raise RuntimeError('additional quota allocation changed')
    return quotas

def _dense_stack(reference, offsets, valid, source_depth):
    from types import SimpleNamespace
    from raptor_global_stack_smart336 import FixedGlobalStack
    if tuple(reference.slot_widths) != tuple(INFERRED96_WIDTHS):
        raise ValueError('capacity-aware sampling is pinned to the Global96 reference layout')
    pools = []
    audit = []
    begin = 0
    for slot, width in enumerate(reference.slot_widths, 1):
        end = begin + width
        old = reference.global_indices[begin:end]
        row = int(reference.source_series_rows[begin])
        info = {'slot': slot, 'original_width': width, 'source_row': row}
        audit.append(info)
        if row < 0 or bool((old < 0).all()):
            pools.append((np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)))
            begin = end
            continue
        start, stop = map(int, offsets[row:row + 2])
        if not 0 <= start < stop <= len(valid) == len(source_depth):
            raise ValueError('source offset drift')
        depth = np.asarray(source_depth[start:stop], dtype=np.float32)
        keep = np.asarray(valid[start:stop], dtype=bool) & np.isfinite(depth) & (depth >= 0.02) & (depth <= 0.98)
        eligible = np.flatnonzero(keep).astype(np.int64) + start
        reverse = bool(old[0] > old[-1] or (old[0] == old[-1] and np.float32(source_depth[old[0]]) != reference.canonical_depths[begin]))
        if reverse:
            eligible = eligible[::-1].copy()
        depth = np.asarray(source_depth[eligible], dtype=np.float32)
        if reverse:
            depth = (1.0 - depth).astype(np.float32)
        pools.append((eligible, depth))
        begin = end
    quotas = _dense_quotas([len(pool[0]) for pool in pools])
    if int(quotas.sum()) < 3:
        raise ValueError('study has fewer than three usable unique source slices')
    parts = {key: [] for key in ('global_indices', 'nominal_slots', 'canonical_depths', 'nominal_steps', 'source_series_rows', 'source_slot_ids')}
    for slot, ((eligible, depth), quota, info) in enumerate(zip(pools, quotas, audit), 1):
        quota = int(quota)
        info.update(eligible=len(eligible), allocated_width=quota, additional=max(0, quota - info['original_width']))
        info['reason'] = 'missing_no_padding' if not quota else 'short_slot_all_unique' if len(eligible) < info['original_width'] else 'uniform_native_centers' if info['additional'] else 'uniform_native_centers'
        positions = np.linspace(0, len(eligible) - 1, quota).round().astype(np.int64)
        if len(positions) != quota or not bool((np.diff(positions) > 0).all()):
            raise RuntimeError('capacity-aware sampling slot selection repeated a source')
        mask = reference.nominal_slots == slot
        source_slot = int(reference.source_slot_ids[mask][0])
        parts['global_indices'].append(eligible[positions])
        parts['canonical_depths'].append(depth[positions])
        parts['nominal_slots'].append(np.full(quota, slot, dtype=np.int8))
        parts['nominal_steps'].append(np.full(quota, 0.96 / max(quota - 1, 1), dtype=np.float32))
        parts['source_series_rows'].append(np.full(quota, info['source_row'], dtype=np.int32))
        parts['source_slot_ids'].append(np.full(quota, source_slot, dtype=np.int8))
    result = FixedGlobalStack(**{key: np.concatenate(value) for key, value in parts.items()}, slot_widths=tuple((int(value) for value in quotas)))
    if len(np.unique(result.global_indices)) != len(result.global_indices) or bool((result.global_indices < 0).any()):
        raise RuntimeError('capacity-aware sampling output contains a repeated/missing source')
    return (result, audit)