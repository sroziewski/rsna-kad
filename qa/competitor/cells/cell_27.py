_a5_sub = pd.read_csv('/kaggle/working/_pipeline_stage.csv', dtype={'StudyInstanceUID':str})
assert _a5_sub.columns.tolist()[1:] == A5_LABELS, 'submission schema drift'
if A5_W > 0:
    _a5_uid_order = _a5_sub.StudyInstanceUID.astype(str).tolist()
    _a5_ours = np.stack([A5_PREDS[u] for u in _a5_uid_order])
    _a5_presence = np.asarray([A5_AVAILABLE[u] for u in _a5_uid_order], dtype=bool)
    _a5_sub[A5_LABELS] = _a5_blend_available(
        _a5_sub[A5_LABELS].to_numpy(), _a5_ours, _a5_presence, A5_W)
    assert np.isfinite(_a5_sub[A5_LABELS].to_numpy()).all()
    _a5_sub.to_csv('/kaggle/working/_pipeline_stage.csv', index=False)