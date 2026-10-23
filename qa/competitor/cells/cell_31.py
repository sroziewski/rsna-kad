# Publish submission.csv only after the entire BTKD + CoAt family graph succeeds.
import platform
_release_path=Path('/kaggle/working/_pipeline_stage.csv')
_release_df=rsna_frame(pd.read_csv(_release_path,dtype={'StudyInstanceUID':str}),_RSNA_TEST_IDS,_RSNA_LABELS,'FINAL')
if _DINOV2_MATCHED_MEMBERS!=20:raise RuntimeError('final DINO member count mismatch')
if not globals().get('V18_CALIBRATOR_APPLIED'):raise RuntimeError('BTKD calibrator was not applied')
_expected_preparations={(name,uid) for name in ('maxspan-v5','native384dense-v10','native384-v8') for uid in _RSNA_TEST_IDS}
if _ke_input_ids != _expected_preparations:
    rsna_event('raptor_preparation_identities_incomplete', missing=len(_expected_preparations - _ke_input_ids), extra=len(_ke_input_ids - _expected_preparations))
rsna_save_predictions('final',_RSNA_TEST_IDS,_release_df[_RSNA_LABELS].to_numpy(),_RSNA_LABELS)
_RSNA_AUDIT.update(status='COMPLETE',dino_members=20,a5_folds=5,raptor_views=4,coats={'resgated_epochs':[4,6,8],'d4_parent_swa_epochs':[15,16],'d4_adapter_swa_epochs':[11,9,5]},family_mix=[.5,.5],inner_public_coat_mix=[.6,.4],inner_rerank=False,outer_weights=_coatnet_weight,bt_input_sha256='eb9c51cf04ddbd4e923f1a5c278e7b019036db9e67ecc20bea6661bd536bb303',v555_input_sha256='22d71a1311ef8282e26a0ca498bed57100da5152352deae7cbdaeca4ab8539c9',elapsed_seconds=time.time()-T0,score_recovery_claimed=False)
for cache_file in list(_RSNA_CACHE_FILES):Path(cache_file).unlink(missing_ok=True)
_final=Path('/kaggle/working/submission.csv');_tmp=_final.with_suffix('.csv.tmp')
with _tmp.open('w') as _out_handle:
    _release_df.to_csv(_out_handle,index=False)
    _out_handle.flush();os.fsync(_out_handle.fileno())
_RSNA_AUDIT['submission_sha256']=rsna_sha(_tmp)
os.replace(_tmp,_final)
rsna_json('/kaggle/working/btkd_v559_complete.json',_RSNA_AUDIT)
print(f'COMPLETE: {_final}, {len(_release_df)} studies, SHA256={_RSNA_AUDIT["submission_sha256"]}')