ORDER_TAGS = [(32, 50), (32, 55), (32, 19)]
DECODE_FAILED = []


def _natural_key(name):
    return tuple((int(x) if x.isdigit() else x.lower() for x in re.split('(\\d+)', str(name))))

def _order_dominant_axis(rec):
    files, d = (rec['files'], rec['dir'])
    rows = []
    for pos, f in enumerate(files):
        ipp = inst = None
        try:
            ds = pydicom.dcmread(os.path.join(d, f), force=True, stop_before_pixels=True, specific_tags=['ImagePositionPatient', 'InstanceNumber'])
            raw = getattr(ds, 'ImagePositionPatient', None)
            if raw is not None and len(raw) >= 3:
                c = np.asarray(raw[:3], dtype=np.float64)
                if np.isfinite(c).all():
                    ipp = c
            n = getattr(ds, 'InstanceNumber', None)
            if n is not None:
                inst = float(n)
        except Exception:
            pass
        rows.append((f, ipp, inst, pos))
    placed = [r for r in rows if r[1] is not None]
    need = max(2, int(0.8 * len(rows)))
    if len(placed) >= need:
        xyz = np.stack([r[1] for r in placed])
        axis = int(np.argmax(np.ptp(xyz, axis=0)))
        spare = float(np.nanmedian(xyz[:, axis]))
        rows.sort(key=lambda r: (float(r[1][axis]) if r[1] is not None else spare, r[2] if r[2] is not None else float('inf'), r[3]))
    elif sum((r[2] is not None for r in rows)) >= need:
        rows.sort(key=lambda r: (r[2] if r[2] is not None else float('inf'), r[3]))
    else:
        rows.sort(key=lambda r: _natural_key(r[0]))
    return ([r[0] for r in rows], True)

def order_slices(rec):
    if RULES['order'] == 'dominant_axis':
        return _order_dominant_axis(rec)
    files, d = (rec['files'], rec['dir'])
    keyed = []
    for f in files:
        k = None
        ds = None
        try:
            ds = pydicom.dcmread(os.path.join(d, f), force=True, stop_before_pixels=True, specific_tags=ORDER_TAGS)
            iop = np.asarray(ds.ImageOrientationPatient, dtype=float)
            ipp = np.asarray(ds.ImagePositionPatient, dtype=float)
            k = float(np.dot(ipp, np.cross(iop[:3], iop[3:])))
        except Exception:
            try:
                k = float(ds.InstanceNumber)
            except Exception:
                k = None
        keyed.append((k, f))
    if any((k is None for k, _ in keyed)):
        return (sorted(files,key=_natural_key), False)
    return ([f for _, f in sorted(keyed, key=lambda t: t[0])], True)

def read_slot(rec, n_slice=None, out_size=None):
    """Same healthy pixels as BTKD; failed slices cannot erase good native planes."""
    n_slice = GROUP if n_slice is None else int(n_slice)
    out_size = IMG if out_size is None else int(out_size)
    files, d, px = rec.get('ordered') or rec['files'], rec['dir'], rec['px']
    if not files:
        rsna_event('slot_no_files', series=str(rec.get('SeriesInstanceUID', d)), dir=str(d))
        return None
    lo, hi = int(SLICE_BAND[0]*(len(files)-1)), int(SLICE_BAND[1]*(len(files)-1))
    idx = np.unique(np.linspace(lo,hi,n_slice).astype(int)) if hi>lo else np.array([len(files)//2])
    while len(idx)<n_slice: idx=np.append(idx,idx[-1])
    decoded, errors = {}, []
    for i in sorted(set(int(v) for v in idx[:n_slice])):
        path=os.path.join(d,files[i])
        try:
            ds=pydicom.dcmread(path,force=True)
            a=ds.pixel_array.astype(np.float32)
            a=a*float(getattr(ds,'RescaleSlope',1) or 1)+float(getattr(ds,'RescaleIntercept',0) or 0)
            if a.ndim!=2 or not np.isfinite(a).all(): raise ValueError('expected finite 2-D pixels')
            decoded[i]=a
        except Exception as exc:
            decoded[i]=None; errors.append({'path':path,'error':f'{type(exc).__name__}: {exc}'})
    planes=[decoded[int(i)] for i in idx[:n_slice]]
    got=[i for i,p in enumerate(planes) if p is not None]
    if errors:
        DECODE_FAILED.append(rec.get('SeriesInstanceUID',d))
        rsna_event('partial_slice_decode',series=str(rec.get('SeriesInstanceUID',d)),errors=errors)
    if not got:
        rsna_event('slot_all_decode_failed', series=str(rec.get('SeriesInstanceUID', d)), dir=str(d), errors=errors[:5])
        return None
    from collections import Counter
    shape=Counter(planes[i].shape for i in got).most_common(1)[0][0]
    odd=[i for i in got if planes[i].shape!=shape]
    if odd:
        rsna_event('slot_shape_mismatch', series=str(rec.get('SeriesInstanceUID', d)), dir=str(d), majority=[int(v) for v in shape], dropped=len(odd))
        for i in odd: planes[i]=None
        got=[i for i in got if planes[i] is not None]
    for i,p in enumerate(planes):
        if p is None:
            planes[i]=np.zeros(shape,np.float32) if RULES['decode_fill']=='zero' else planes[min(got,key=lambda j:abs(j-i))]
    vol=np.stack(planes)
    if px and np.isfinite(px) and px>0:
        want=int(round(CROP_MM/px)); h,w=shape
        if 16<want<min(h,w):
            cy,cx=h//2,w//2; half=want//2
            vol=vol[:,max(0,cy-half):cy+half,max(0,cx-half):cx+half]
    lo_v,hi_v=np.percentile(vol,[1,99])
    vol=np.clip((vol-lo_v)/max(hi_v-lo_v,1e-6),0,1)
    t=torch.from_numpy(np.ascontiguousarray(vol)).unsqueeze(0)
    t=F.interpolate(t,size=(out_size,out_size),mode='bilinear',align_corners=False)
    return (t.squeeze(0)*255).round().clamp(0,255).to(torch.uint8)