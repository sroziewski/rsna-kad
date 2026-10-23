SERIES_ROOT = COMP / 'test_series'
if not SERIES_ROOT.is_dir():
    raise FileNotFoundError('Missing test_series; train-series fallback is forbidden')
print('series root:', SERIES_ROOT)

def ordered_files(sdir, cap=64):
    # Preserve the parent's InstanceNumber ordering; remove pre-sort truncation.
    # The unused cap argument remains only for call compatibility.
    keyed=[]; unreadable=[]
    for f in sorted(sdir.glob('*.dcm')):
        try:
            ds=pydicom.dcmread(str(f),stop_before_pixels=True)
            key=int(ds.InstanceNumber)
        except Exception as exc:
            unreadable.append(str(f)); continue
        keyed.append((key,str(f)))
    if unreadable:
        rsna_event('a5_order_unreadable_dropped',series=str(sdir),unreadable=len(unreadable),readable=len(keyed),examples=unreadable[:3])
    return [f for _,f in sorted(keyed)]

def series_side(path):
    try:
        return float(pydicom.dcmread(path, stop_before_pixels=True).ImagePositionPatient[0])
    except Exception:
        return 0.0

def read_crop(path):
    try:
        ds = pydicom.dcmread(path)
        arr = ds.pixel_array.astype(np.float32)
    except Exception:
        return None
    try:
        ps = float(ds.PixelSpacing[0])
    except Exception:
        ps = CROP_MM / max(arr.shape)
    half = int(round(CROP_MM / ps / 2))
    cy, cx = (arr.shape[0] // 2, arr.shape[1] // 2)
    y0, y1 = (max(0, cy - half), min(arr.shape[0], cy + half))
    x0, x1 = (max(0, cx - half), min(arr.shape[1], cx + half))
    crop = arr[y0:y1, x0:x1]
    return None if crop.size == 0 else crop

def window(crop, lo, hi, flip):
    c = np.clip((crop - lo) / max(hi - lo, 1e-06), 0, 1)
    img = cv2.resize(c, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    return img[:, ::-1].copy() if flip else img

def render(path, flip):
    crop = read_crop(path)
    if crop is None:
        return None
    lo, hi = np.percentile(crop[::4, ::4], [1, 99])
    return window(crop, lo, hi, flip)

def build_study(args):
    idx,study,recs=args
    out=np.zeros((N_SLOT,N_SLICE,SIZE,SIZE),np.uint8);mask=np.zeros(N_SLOT,np.uint8)
    rows=pd.DataFrame(recs)
    if not len(rows):
        rsna_event('a5_no_metadata',study=str(study)); return idx,out,mask
    for s_i,(plane,fs) in enumerate(SLOTS):
        sub=rows[(rows.Anatomical_Plane==plane)&(rows.Fat_Suppression==fs)]
        if sub.empty: continue  # Actual acquisition absence, not a decoding failure.
        series=str(sub.iloc[0].SeriesInstanceUID)
        files=ordered_files(SERIES_ROOT/str(study)/series)
        if not files:
            rsna_event('a5_slot_no_files',study=str(study),series=series); continue
        flip=plane!='Sagittal' and series_side(files[0])<0
        i0=int(round(SLICE_BAND[0]*(len(files)-1)));i1=int(round(SLICE_BAND[1]*(len(files)-1)))
        avail=list(range(i0,i1+1))
        if len(avail)>=N_SLICE:
            picks=[avail[int(round(t))] for t in np.linspace(0,len(avail)-1,N_SLICE)];off=0
        else:picks,off=avail,(N_SLICE-len(avail))//2
        rendered=0;errors=[]
        if INTENSITY=='series':
            crops=[read_crop(files[p]) for p in picks];got=[x for x in crops if x is not None]
            if not got:
                rsna_event('a5_slot_all_crops_failed',study=str(study),series=series); continue
            samp=np.concatenate([x[::4,::4].ravel() for x in got]);lo_,hi_=np.percentile(samp,[1,99])
            for c,x in enumerate(crops):
                if x is None:x=read_crop(files[min(len(files)-1,picks[c]+1)])
                if x is None:errors.append(files[picks[c]]);continue
                img=window(x,lo_,hi_,flip)
                rsna_finite(img,'A5 rendered pixels')
                out[s_i,off+c]=(img*255).astype(np.uint8);rendered+=1
        else:
            for c,p in enumerate(picks):
                img=render(files[p],flip)
                if img is None:img=render(files[min(len(files)-1,p+1)],flip)
                if img is None:errors.append(files[p]);continue
                rsna_finite(img,'A5 rendered pixels')
                out[s_i,off+c]=(img*255).astype(np.uint8);rendered+=1
        if errors or rendered!=len(picks):
            rsna_event('a5_decode_failure',study=str(study),series=series,failed=errors,rendered=int(rendered),wanted=int(len(picks)))
            if not rendered: continue
        mask[s_i]=rendered
    if not mask.any():
        rsna_event('a5_empty_study',study=str(study))
        print(f'[a5] {study}: no acquired slots; all-zero mask kept (parent policy)', flush=True)
    return idx,out,mask
sub_df = pd.read_csv(COMP / 'sample_submission.csv',dtype={'StudyInstanceUID':str})
ser_csv = pd.read_csv(COMP / 'test_series.csv',dtype={'StudyInstanceUID':str,'SeriesInstanceUID':str})
if set(sub_df.StudyInstanceUID)!=set(_RSNA_TEST_IDS):
    raise RuntimeError('A5 test/sample UID mismatch')
ser_csv = ser_csv.loc[:, ~ser_csv.columns.duplicated()]
studies = sub_df.StudyInstanceUID.tolist()
by = {s: g.to_dict('records') for s, g in ser_csv[ser_csv.StudyInstanceUID.isin(set(studies))].groupby('StudyInstanceUID')}
print(f'{len(studies):,} test studies, {len(by):,} with series metadata')