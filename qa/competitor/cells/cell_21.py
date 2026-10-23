def write_submission(pred, studies, test_df, path):
    sub = pd.DataFrame(pd.DataFrame(pred).rank(pct=True).values, columns=TARGETS)
    sub.insert(0, 'StudyInstanceUID', studies)
    sub = test_df[['StudyInstanceUID']].merge(sub, on='StudyInstanceUID', how='left')
    if sub[TARGETS].isna().any().any():
        rsna_event('submission_neutral_fill', studies=int(sub[TARGETS].isna().any(axis=1).sum()))
        sub[TARGETS] = sub[TARGETS].fillna(0.5)
    sub.to_csv(path, index=False)
    return sub

def _v37_validate_submission(path, test_df, tag):
    frame = pd.read_csv(path)
    expected = ['StudyInstanceUID'] + TARGETS
    if list(frame.columns) != expected:
        raise ValueError(f'{tag}: columns differ from the competition contract')
    if len(frame) != len(test_df) or not frame['StudyInstanceUID'].is_unique:
        raise ValueError(f'{tag}: row count or StudyInstanceUID uniqueness failed')
    if set(frame['StudyInstanceUID'].astype(str)) != set(test_df['StudyInstanceUID'].astype(str)):
        raise ValueError(f'{tag}: StudyInstanceUID set differs from test.csv')
    values = frame[TARGETS].to_numpy(np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f'{tag}: non-finite prediction')
    return frame

def main():
    pkg = find_weights()
    if pkg is None:
        raise WeightsError('required public checkpoint manifest not found')
    infer_from_package(pkg, DEVS[0])
    _v37_validate_submission('_pipeline_stage.csv', pd.read_csv(ROOT / 'test.csv'), 'public frontier')