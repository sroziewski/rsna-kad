def log(msg):
    print(f'[{time.time() - T0:7.1f}s] {msg}', flush=True)

def _one_direct(tag, candidates, required):
    matches = []
    for candidate in map(Path, candidates):
        if candidate.is_dir() and all((candidate / item).exists() for item in required):
            matches.append(candidate)
    unique = []
    for match in matches:
        if str(match.resolve()) not in {str(item.resolve()) for item in unique}:
            unique.append(match)
    if len(unique) != 1:
        raise FileNotFoundError(
            f'{tag}: expected exactly one direct mounted artifact, found '
            f'{[str(item) for item in unique]}; checked {[str(Path(item)) for item in candidates]}'
        )
    return unique[0]

def find_root():
    return _one_direct('competition', [
        os.environ.get('RSNA_COMP_ROOT', '/kaggle/input/competitions/rsna-knee-abnormality-detection'),
        '/kaggle/input/rsna-knee-abnormality-detection',
    ], ['test.csv', 'test_series.csv', 'test_series'])

def find_dinov2(variant='small'):
    if variant != 'small':
        raise ValueError(f'unpinned DINOv2 variant: {variant}')
    return _one_direct('DINOv2-S model', [
        '/kaggle/input/models/metaresearch/dinov2/PyTorch/small/1',
        '/kaggle/input/models/metaresearch/dinov2/pytorch/small/1',
        '/kaggle/input/dinov2/PyTorch/small/1',
        '/kaggle/input/dinov2/pytorch/small/1',
    ], ['config.json', 'pytorch_model.bin'])
ROOT = find_root()
DINOV2_SOURCE = find_dinov2('small')
log(f'input root: {ROOT}')
_STARTUP_PATH_S = _startup_time.perf_counter() - _STARTUP_T0 - _STARTUP_GPU_IMPORT_S
print(f'[startup] competition-root preflight: {_STARTUP_PATH_S:.2f}s', flush=True)
IMG = CACHE_IMG

def available_gb():
    try:
        with open('/proc/meminfo') as fh:
            info = {k.strip(): v for k, v in (l.split(':', 1) for l in fh if ':' in l)}
        return int(info['MemAvailable'].split()[0]) / 1024 ** 2
    except Exception:
        return CACHE_BUDGET_GB / CACHE_FRACTION

def plan_cache(n_study, n_test=0):
    avail = available_gb()
    budget = min(avail * CACHE_FRACTION, CACHE_BUDGET_MAX_GB)
    n_total = n_study + max(n_test, int(TEST_SHARE * n_study))
    per_slice = n_total * N_SLOT * IMG * IMG
    afford = int(budget * 1024 ** 3 // max(per_slice, 1))
    groups = max(1, min(N_GROUP_MAX, afford // GROUP))
    log(f'memory: {avail:.1f} GB available, {budget:.1f} GB to the cache; sizing for {n_study} train + {n_total - n_study} test studies -> {groups} group(s) of {GROUP} = {groups * GROUP} slices per slot' + (f' (wanted {N_GROUP_MAX})' if groups < N_GROUP_MAX else ''))
    return groups
N_GROUP = plan_cache(len(pd.read_csv(ROOT / 'train.csv')), len(pd.read_csv(ROOT / 'test.csv')))
CACHE_SLICES = GROUP * N_GROUP
log(f'cache layout: {N_GROUP} groups x {GROUP} slices = {CACHE_SLICES} per slot')
rsna_initialize(pd.read_csv(ROOT/'test.csv',dtype={'StudyInstanceUID':str}).StudyInstanceUID.tolist())
os.environ['RSNA_COMP_ROOT']=str(ROOT)
rsna_asset_preflight()

# Resolve unsupported A5 metadata before long model inference, without inventing labels.
_meta = pd.read_csv(ROOT/'test_series.csv', dtype={'StudyInstanceUID':str})
_required = {'StudyInstanceUID','Anatomical_Plane','Fat_Suppression'}
if not _required.issubset(_meta.columns):
    raise RuntimeError('Missing required series metadata columns: '+str(sorted(_required-set(_meta.columns))))
_a5_matches = _meta.Anatomical_Plane.isin(['Sagittal','Coronal','Axial']) & _meta.Fat_Suppression.isin([0,1])
_a5_ids = set(_meta.loc[_a5_matches,'StudyInstanceUID'])
_a5_unavailable = [u for u in _RSNA_TEST_IDS if u not in _a5_ids]
rsna_json('/kaggle/working/diagnostics/a5_metadata_applicability.json', {
    'studies':len(_RSNA_TEST_IDS), 'no_matching_acquisition_uids':_a5_unavailable,
    'policy':'base-only contribution for explicit A5 absence; model failures remain fatal'})
rsna_phase('input_metadata', 'COMPLETE', a5_without_matching_acquisition=len(_a5_unavailable),
           missing_fluid_flags=int(_meta.Fluid_Sensitive.isna().sum()) if 'Fluid_Sensitive' in _meta else len(_meta))
del _meta, _a5_matches, _a5_ids, _a5_unavailable