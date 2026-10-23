import os
for _thread_env in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_thread_env, '4')
os.environ.setdefault('CUDNN_CONV_WSCAP_DBG', '1024')

def _d4_check_runtime(rt, artifact_root):
    import hashlib
    import json
    from pathlib import Path
    import torch
    root = Path(artifact_root)
    manifest_path = root / 'coatnet_pairfilm_manifest.json'
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if digest != '7ada0605bca6b0530569c6454e988ace479606a3328ed591d090e5764fea661d':
        raise RuntimeError('D4 reference manifest identity changed')
    expected = json.loads(manifest_path.read_text())
    manifest, paths = rt.validate_artifact_root(root, verify_hashes=True)
    if manifest != expected or len(paths) != 1 or rt.N_MODELS != 1:
        raise RuntimeError('D4 must load exactly the reference SWA model')
    if Path(paths[0]).name != 'coatnet_d4_parent_rank8_swa2.pt':
        raise RuntimeError('D4 must use its reference rank-8 parent checkpoint')
    if manifest['slice_depth']['adapter_sha256'] != 'b7c48b19997bc7ecb7e997f8dcf491ec5701ff917d1dac4b369eb85ac97ae89a':
        raise RuntimeError('D4 reference adapter identity changed')
    if rt.GPU_BATCH_STUDIES != 2 or rt.BACKBONE_MICRO_IMAGES != 8:
        raise RuntimeError('D4 reference batching changed')
    if rt.CUDNN_BENCHMARK is not False:
        raise RuntimeError('D4 reference convolution policy changed')
    if rt.base.cv2.__version__ != '4.12.0' or rt.parent.timm.__version__ != '1.0.22':
        raise RuntimeError('D4 reference dependency versions changed')
    if torch.cuda.device_count() != 2:
        raise RuntimeError('D4 requires exactly two T4 GPUs')
    names = [torch.cuda.get_device_name(i) for i in range(2)]
    if not all(('T4' in name for name in names)):
        raise RuntimeError(f'D4 reference device mismatch: {names}')
    return (manifest, paths)

def _d4_check_outputs(rt, manifest, receipt, output, competition):
    import hashlib
    from pathlib import Path
    import numpy as np
    import pandas as pd

    def need(condition, message):
        if not condition:
            raise RuntimeError('D4 release check: ' + message)
    need(receipt['status'] == rt.SUBMISSION_STATUS, 'unsuccessful status')
    expected_fields = {'models': 1, 'fallback_studies': 0, 'dicom_preparations_per_study': 1, 'model_passes_per_study': 1, 'rank_after_probability_average': True, 'canonical_sagittal': True, 'common_physical_triplet_warp': False, 'resolution': 384, 'slice_depth_arm': 'd4_zones', 'slice_depth_state_elements': 3255, 'slice_depth_parent_checkpoint_sha256': '5de38e333e3184d7b52fb068183f0cc198d6cbabb6ffbe36b26338622e82d297', 'fsx_attention_heads': 2, 'fsx_attention_width': 128, 'fsx_serving_delta_cap': 0.3, 'fsx_training_delta_cap': 0.5, 'head': 'rank8_swa2_plus_d4_three_zone_expanded_fsx', 'slice_depth_swa_member_epochs': [11, 9, 5], 'slice_depth_checkpoint_sha256': manifest['slice_depth']['adapter_sha256']}
    for key, value in expected_fields.items():
        need(receipt[key] == value, key)
    need(1 <= int(receipt['maximum_eval_windows']) <= 94, 'window limit')
    checkpoints = receipt['checkpoints']
    need(len(checkpoints) == 1, 'parent checkpoint count')
    need(checkpoints[0]['sha256'] == '5de38e333e3184d7b52fb068183f0cc198d6cbabb6ffbe36b26338622e82d297', 'parent checkpoint hash')
    need(checkpoints[0]['member_epochs'] == [15, 16], 'parent SWA member epochs')
    need(receipt['slice_depth_checkpoint_sha256'] == 'b7c48b19997bc7ecb7e997f8dcf491ec5701ff917d1dac4b369eb85ac97ae89a', 'adapter checkpoint hash')
    processes = receipt['processes']
    need(len(processes) == 2 and all((int(p['returncode']) == 0 for p in processes)), 'worker process failure')
    shards = receipt['shards']
    need(len(shards) == 2, 'shard count')
    for s in shards:
        need(s['device'] == 'cuda', 'non-CUDA shard')
        need(int(s['fallback_studies']) == 0 and int(s['preparation_warnings']) == 0, 'failed study/preparation')
        need(int(s['models_resident']) == 1, 'resident model count')
        need(int(s['peak_reserved_bytes']) < int(15.5 * 1024 ** 3), 'reference memory gate')
        need(s['specialized_worker_loader_contract'] == manifest['worker_loader_contract'], 'specialized loader')
    output = Path(output)
    prediction_path = output.parent / rt.PREDICTIONS_NAME
    with np.load(prediction_path, allow_pickle=False) as payload:
        uids = payload['study_uids'].astype(str).tolist()
        raw = np.asarray(payload['raw_probabilities'], dtype=np.float32)
        mean = np.asarray(payload['probability_mean'], dtype=np.float32)
        ranked = np.asarray(payload['submission_percentile_rank'], dtype=np.float64)
    ids = pd.read_csv(Path(competition) / 'test.csv', dtype={'StudyInstanceUID': str}).StudyInstanceUID.tolist()
    need(len(ids) > 0 and len(ids) == len(set(ids)), 'test identity')
    need(uids == ids and int(receipt['studies']) == len(ids), 'prediction identity/count')
    need(raw.shape == (1, len(ids), 12), 'raw probability shape')
    need(np.isfinite(raw).all() and ((raw >= 0) & (raw <= 1)).all(), 'raw probabilities')
    need(np.array_equal(raw.astype(np.float64).mean(0).astype(np.float32), mean), 'probability mean')
    need(np.array_equal(rt.percentile_rank64(mean), ranked), 'global percentile ranks')
    frame = pd.read_csv(output, dtype={'StudyInstanceUID': str}, float_precision='round_trip')
    need(frame.columns.tolist() == ['StudyInstanceUID', *rt.base.LABELS], 'CSV labels')
    need(frame.StudyInstanceUID.tolist() == ids, 'CSV identity')
    need(np.array_equal(frame.iloc[:, 1:].to_numpy(np.float64), ranked), 'CSV numeric round trip')
    actual_hash = hashlib.sha256(output.read_bytes()).hexdigest()
    need(actual_hash == receipt['output_sha256'], 'CSV hash')
    receipt['reference_wrapper_checks'] = {'export_sha256': '7f8410fdf1b60cd3831d50f517bb25798a4e4a3404c59dc269193d5fe7cce729', 'validated_manifest_sha256': '7ada0605bca6b0530569c6454e988ace479606a3328ed591d090e5764fea661d', 'probability_mean_exact': True, 'ranks_exact': True, 'csv_roundtrip_exact': True, 'source_and_model_hashes_validated': True, 'input_identity_equals_original_grid': False}
    return receipt

def _run_required_child(command, environment, logfile):
    import os
    import signal
    import subprocess
    import time
    from pathlib import Path
    remaining = TIME_BUDGET - (time.time() - T0)
    if remaining <= 0:
        raise TimeoutError('No time remains for a required model branch')
    path = Path(logfile)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as log_handle:
        process = subprocess.Popen(command, env=environment, stdout=log_handle, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            returncode = process.wait(timeout=remaining)
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            raise
    if returncode != 0:
        with path.open('rb') as handle:
            handle.seek(max(0, path.stat().st_size - 5000))
            tail = handle.read().decode('utf-8', errors='replace')
        raise RuntimeError(f'Required model branch failed ({returncode}); {path}\n{tail}')
    return returncode
ASSET_ROOTS = ['/kaggle/input/datasets/dreaddevelopment/raptor-knee-maxspan', '/kaggle/input/raptor-knee-maxspan', '/kaggle/input/datasets/dreaddevelopment/raptor-knee-native384', '/kaggle/input/raptor-knee-native384', '/kaggle/input/datasets/dreaddevelopment/raptor-knee-native384dense', '/kaggle/input/raptor-knee-native384dense', '/kaggle/input/datasets/mattiaangeli/knee-mri-fold-weights', '/kaggle/input/knee-mri-fold-weights', '/kaggle/input/datasets/mattiaangeli/opencv-python-headless-4120088-x86', '/kaggle/input/opencv-python-headless-4120088-x86', '/kaggle/input/datasets/marwanmath/resnet-50-radimagenet-marwan', '/kaggle/input/resnet-50-radimagenet-marwan', '/kaggle/input/datasets/mattiaangeli/rsna-knee-coat-resgated-ep10-top3', '/kaggle/input/rsna-knee-coat-resgated-ep10-top3', '/kaggle/input/datasets/antoinegg1/rsna-knee-e11-diverse-heads-v20', '/kaggle/input/rsna-knee-e11-diverse-heads-v20', '/kaggle/input/datasets/antoinegg1/rsna-knee-e9-radimagenet-heads-v15', '/kaggle/input/rsna-knee-e9-radimagenet-heads-v15', '/kaggle/input/datasets/pilkwang/rsna-knee-llm-labels', '/kaggle/input/rsna-knee-llm-labels', '/kaggle/input/datasets/prvsiyan/rsna-knee-v52-radimagenet-heads-20260812', '/kaggle/input/rsna-knee-v52-radimagenet-heads-20260812', '/kaggle/input/datasets/pilkwang/rsna-knee-weights', '/kaggle/input/rsna-knee-weights', '/kaggle/input/datasets/mattiaangeli/rsna-knee-coatnet-d4-depthzone-swa3-b2', '/kaggle/input/rsna-knee-coatnet-d4-depthzone-swa3-b2', '/kaggle/input/notebooks/sofiaanjenje/rsna-knee-e11-train', '/kaggle/input/rsna-knee-e11-train', '/kaggle/input/notebooks/sofiaanjenje/rsna-knee-e13-train', '/kaggle/input/rsna-knee-e13-train', '/kaggle/input/models/metaresearch/dinov2/pytorch/small/1', '/kaggle/input/dinov2/pytorch/small/1']

def _asset_walk():
    from pathlib import Path
    seen = set()
    for root in ASSET_ROOTS:
        root = Path(root).resolve()
        if not root.is_dir():
            continue
        level = [root]
        for _depth in range(6):
            following = []
            for directory in sorted(level):
                if str(directory) in seen:
                    continue
                seen.add(str(directory))
                children = sorted(directory.iterdir())
                dirs = [p for p in children if p.is_dir() and p.name not in ('train_series', 'test_series', 'train_images', 'test_images')]
                files = [p.name for p in children if p.is_file()]
                yield (str(directory), [p.name for p in dirs], files)
                following.extend(dirs)
            level = following

def _asset_find_asset(name, digest=None):
    import fnmatch, hashlib
    from pathlib import Path
    hits = []
    for root, _, files in _asset_walk():
        for file in files:
            if fnmatch.fnmatchcase(file, name):
                p = Path(root) / file
                if digest:
                    h = hashlib.sha256()
                    with p.open('rb') as handle:
                        for block in iter(lambda: handle.read(8 << 20), b''):
                            h.update(block)
                    if h.hexdigest() != digest:
                        continue
                if p.resolve() not in [h.resolve() for h in hits]:
                    hits.append(p)
    if len(hits) != 1:
        raise RuntimeError(f'expected one pinned asset {name}, got {hits}')
    return hits[0]

def _asset_half_rank_mix(top3, swa3):
    if top3.shape != swa3.shape or not top3.index.equals(swa3.index):
        raise ValueError('CoAt family alignment changed')
    if top3.isna().any().any() or swa3.isna().any().any():
        raise ValueError('missing CoAt family prediction')
    units = top3.rank(method='average') + swa3.rank(method='average')
    return units.rank(method='average', pct=True)

def _asset_write_coat_runtime(artifact, output, helper_source):
    from pathlib import Path
    import hashlib
    source = Path(artifact) / 'coatnet_resgated_ep10_top3_inference.py'
    text = source.read_text()
    if hashlib.sha256(source.read_bytes()).hexdigest() != 'b11e58f8d7cabe9e264ac70b01e811dc6a846aa01248ace1f0e6536d0d4094d9':
        raise RuntimeError('unexpected parent CoAt runtime')
    edits = {'if not 1 <= windows <= MAX_EVAL_WINDOWS:': 'if not 1 <= windows <= 94:', 'or int(counts.max()) > MAX_EVAL_WINDOWS:': 'or int(counts.max()) > 94:', '"runtime_contract": RUNTIME_CONTRACT,': '"runtime_contract": RUNTIME_CONTRACT, "input_max_windows": 94,', '"eval_grid": "saved_training_gold_unique_center_slot_budgets",': '"eval_grid": "input_unique_native_centers_six_slot_v1",'}
    for old, new in edits.items():
        if old not in text:
            raise RuntimeError('CoAt patch target missing: ' + old)
        text = text.replace(old, new)
    text = text.replace('"gold58_rank_ensemble_auc": manifest["selection"]', '"historical_parent_gold58_rank_ensemble_auc": manifest["selection"]')
    text = text.replace('"t4_gold58_rank_ensemble_auc": manifest["t4_numerical_portability"]', '"historical_parent_t4_gold58_rank_ensemble_auc": manifest["t4_numerical_portability"]')
    marker = 'if __name__ == "__main__":'
    if text.count(marker) != 1:
        raise RuntimeError('CoAt entrypoint changed')
    patch = helper_source + ('\n_original_train_faithful_eval_specs = train_faithful_eval_specs\n'
                             'def _tolerant_coat_specs(study):\n    try:\n        return _dense_coat_specs(study)\n'
                             '    except Exception as exc:\n        print("[dense-fallback] " + str(getattr(study, "study_uid", "?")) + ": " + type(exc).__name__ + ": " + str(exc), flush=True)\n'
                             '        return _original_train_faithful_eval_specs(study)\n'
                             'train_faithful_eval_specs = _tolerant_coat_specs\nRUNTIME_CONTRACT = "resgated_ep10_input_nativecenters_v1"\n')
    text = text.replace(marker, patch + '\n' + marker)
    compile(text, str(output), 'exec')
    Path(output).write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()

def _asset_patch_d4(rt, grid_source):
    import ast
    import hashlib
    import inspect
    import json
    import os
    import textwrap
    from pathlib import Path
    audited = rt.parent.audited
    original = audited.prepare_global96_bag
    original_shard = rt._PARENT_INFER_SHARD
    if getattr(original, '_input_patch_installed', False):
        raise RuntimeError('D4 input preparation was already patched')
    src = textwrap.dedent(inspect.getsource(original))
    tree = ast.parse(src)
    fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef)), None)
    if fn is None or fn.name != 'prepare_global96_bag':
        raise RuntimeError('D4 preparation function identity changed')
    if 'study_uid' not in inspect.signature(original).parameters:
        raise RuntimeError('D4 preparation study-UID contract changed')
    candidates = [n for n in ast.walk(fn) if isinstance(n, ast.Assign) and any((isinstance(t, ast.Name) and t.id == 'specs' for t in n.targets)) and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute) and isinstance(n.value.func.value, ast.Name) and (n.value.func.value.id == 'global_stack') and (n.value.func.attr == 'sample_global_windows')]
    if len(candidates) != 1 or candidates[0] not in fn.body:
        raise RuntimeError('D4 expected one top-level Global96 specs constructor')
    call = candidates[0]
    keywords = {k.arg: k.value for k in call.value.keywords}
    if not (isinstance(keywords.get('count'), ast.Constant) and keywords['count'].value is None and isinstance(keywords.get('train'), ast.Constant) and (keywords['train'].value is False)):
        raise RuntimeError('D4 original evaluation must use count=None, train=False')
    exec(compile(grid_source, '<input-grid>', 'exec'), audited.__dict__)
    audit_dir = Path(os.environ['RSNA_D4_INPUT_AUDIT_DIR'])
    audit_dir.mkdir(parents=True, exist_ok=True)

    def record_input(study_uid, stack, audit):
        import numpy as np
        arrays = ('global_indices', 'nominal_slots', 'canonical_depths', 'nominal_steps', 'source_series_rows', 'source_slot_ids')
        digest = hashlib.sha256()
        for name in arrays:
            value = np.ascontiguousarray(getattr(stack, name))
            digest.update(name.encode())
            digest.update(str(value.dtype).encode())
            digest.update(str(value.shape).encode())
            digest.update(value.tobytes())
        record = {'study_uid': str(study_uid), 'pid': os.getpid(), 'positions': int(len(stack.global_indices)), 'windows_expected': int(len(stack.global_indices) - 2), 'slot_widths': list(map(int, stack.slot_widths)), 'stack_sidecars_sha256': digest.hexdigest(), 'slots': audit}
        with (audit_dir / f'input_{os.getpid()}.jsonl').open('a') as handle:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + '\n')
    audited.__dict__['_record_input'] = record_input
    injection = ast.parse('try:\n    stack, input_audit = _dense_stack(stack, study.offsets, study.valid, study.source_depth)\n'
                          'except Exception as _dense_exc:\n    print("[dense-fallback] " + str(study_uid) + ": " + type(_dense_exc).__name__ + ": " + str(_dense_exc), flush=True)\n'
                          '    input_audit = [{"fallback": type(_dense_exc).__name__ + ": " + str(_dense_exc)}]\n'
                          '_record_input(study_uid, stack, input_audit)\n').body
    where = fn.body.index(call)
    fn.body[where:where] = injection
    ast.fix_missing_locations(tree)
    exec(compile(tree, '<D4-input-preparation-only>', 'exec'), audited.__dict__)
    patched = audited.prepare_global96_bag
    patched._input_patch_installed = True
    for module in (rt, rt.parent):
        if getattr(module, 'prepare_global96_bag', None) is original:
            module.prepare_global96_bag = patched
    if rt._PARENT_INFER_SHARD is not original_shard:
        raise RuntimeError('D4 parent shard must remain the official callable')
    files = {}
    for name, module in [('d4_runtime', rt), ('parent_runtime', rt.parent), ('audited_runtime', audited)]:
        path = Path(module.__file__)
        if path.is_file():
            files[name] = {'file': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    receipt = {'contract': 'd4-original-runtime-input-preparation-only-v2', 'reference_script_version_id': 348764100, 'reference_export_available': True, 'original_prepare_source_sha256': hashlib.sha256(src.encode()).hexdigest(), 'patched_prepare_source_sha256': hashlib.sha256(ast.unparse(tree).encode()).hexdigest(), 'model_loader_modified': False, 'zone_head_modified': False, 'parent_infer_shard_replaced': False, 'grid_source_sha256': hashlib.sha256(grid_source.encode()).hexdigest(), 'artifact_source_files': files}
    (audit_dir / f'patch_{os.getpid()}.json').write_text(json.dumps(receipt, indent=2))
    return receipt

def _asset_run_d4(artifact, environment_dir, competition, output):
    import hashlib, inspect, json, os, sys
    from pathlib import Path
    import pandas as pd
    artifact, output = (Path(artifact), Path(output))
    manifest = artifact / 'coatnet_pairfilm_manifest.json'
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != '7ada0605bca6b0530569c6454e988ace479606a3328ed591d090e5764fea661d':
        raise RuntimeError('D4 SWA3 manifest changed')
    grid = 'import numpy as np\nfrom raptor_global_stack_smart336 import FixedGlobalStack, INFERRED96_WIDTHS\n' + '\n'.join((inspect.getsource(f) for f in (_dense_quotas, _dense_stack)))
    audit_dir = output.parent / 'study_input_receipts'
    audit_dir.mkdir(parents=True, exist_ok=True)
    for stale in audit_dir.glob('*.json*'):
        stale.unlink()
    output.unlink(missing_ok=True)
    output.with_suffix('.receipt.json').unlink(missing_ok=True)
    wrapper = output.parent / 'd4_worker_entry.py'
    code = f'import sys\nsys.path.insert(0,{str(environment_dir)!r})\nsys.path.insert(0,{str(artifact)!r})\nfrom pathlib import Path\nimport json\nimport coatnet_d4_depthzone_swa_inference as rt\n' + inspect.getsource(_asset_patch_d4) + '\n' + inspect.getsource(_d4_check_runtime) + '\n' + inspect.getsource(_d4_check_outputs) + '\n' + f"if __name__ == '__main__' and '--worker-output' not in sys.argv:\n    _checked_manifest, _checked_paths = _d4_check_runtime(rt, Path({str(artifact)!r}))\n" + f'_asset_patch_d4(rt,{grid!r})\n' + 'rt.parent.audited.__file__ = __file__\n' + "if __name__ == '__main__':\n" + "    if '--worker-output' in sys.argv:\n        raise SystemExit(rt.main())\n" + '    manifest, paths = _checked_manifest, _checked_paths\n' + '    r = rt.run_submission(\n' + f'        competition_root=Path({str(competition)!r}), artifact_root=Path({str(artifact)!r}),\n' + f'        output_path=Path({str(output)!r}), gpu_batch_studies=2, backbone_micro_images=8)\n' + f'    r = _d4_check_outputs(rt, manifest, r, Path({str(output)!r}), Path({str(competition)!r}))\n' + f"    Path({str(output.with_suffix('.receipt.json'))!r}).write_text(json.dumps(r,indent=2,allow_nan=False))\n"
    compile(code, str(wrapper), 'exec')
    wrapper.write_text(code)
    env = dict(os.environ)
    env['RSNA_D4_INPUT_AUDIT_DIR'] = str(audit_dir)
    env['PYTHONPATH'] = f'{environment_dir}:/kaggle/working/_coat_env:{artifact}:' + env.get('PYTHONPATH', '')
    env.pop('CUDNN_CONV_WSCAP_DBG', None)
    _run_required_child([sys.executable, str(wrapper)], env, output.with_suffix('.log'))
    receipt = json.loads(output.with_suffix('.receipt.json').read_text())
    expected = pd.read_csv(Path(competition) / 'test.csv', dtype={'StudyInstanceUID': str}).StudyInstanceUID.tolist()
    records = []
    for path in sorted(audit_dir.glob('input_*.jsonl')):
        records.extend((json.loads(line) for line in path.read_text().splitlines() if line.strip()))
    ids = [r['study_uid'] for r in records]
    if len(ids) != len(expected) or len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise RuntimeError('D4 input hook must execute exactly once for every test study')
    if not all((1 <= r['windows_expected'] <= 94 and r['positions'] == r['windows_expected'] + 2 for r in records)):
        raise RuntimeError('D4 input record cardinality drift')
    receipt['input_coverage'] = {'studies': len(records), 'folder': str(audit_dir), 'min_windows': min((r['windows_expected'] for r in records)), 'max_windows': max((r['windows_expected'] for r in records)), 'original_model_loader_retained': True, 'original_shard_scheduler_retained': True}
    output.with_suffix('.receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False))
    return receipt
_speed_original_asset_walk = _asset_walk
_speed_original_find_asset = _asset_find_asset
'Catalogue named immutable attachments once; no competition-tree search.'
import threading as _speed_threading
_speed_asset_lock = _speed_threading.RLock()
_speed_asset_catalogue = None
_speed_asset_hits = {}

def _asset_walk():
    global _speed_asset_catalogue
    with _speed_asset_lock:
        if _speed_asset_catalogue is None:
            _speed_asset_catalogue = tuple(((root, tuple(dirs), tuple(files)) for root, dirs, files in _speed_original_asset_walk()))
        catalogue = _speed_asset_catalogue
    for root, dirs, files in catalogue:
        yield (root, list(dirs), list(files))

def _asset_find_asset(name, digest=None):
    key = (str(name), digest)
    with _speed_asset_lock:
        if key not in _speed_asset_hits:
            _speed_asset_hits[key] = _speed_original_find_asset(name, digest)
        return _speed_asset_hits[key]