def build_cache(slot_map, plane_map, lat_map, tag):
    studies = list(_RSNA_TEST_IDS) if _RSNA_TEST_IDS is not None else sorted(slot_map)
    if set(studies)!=set(slot_map):
        rsna_event('coverage_mismatch',tag=tag,missing=[str(s) for s in studies if s not in slot_map][:20],extra=len(set(slot_map)-set(studies)))
        slot_map={s:slot_map.get(s,{}) for s in studies}
    sidx={s:i for i,s in enumerate(studies)}
    cache=rsna_array((len(studies),N_SLOT,CACHE_SLICES,IMG,IMG),tag)
    mask=np.zeros((len(studies),N_SLOT),np.float32)
    jobs=[(st,k,plane,slot_map[st][name]) for st in studies for k,(name,plane,_,_) in enumerate(SLOTS) if name in slot_map[st]]
    log(f'{tag}: {len(studies)} studies, {len(jobs)} slot-series, {cache.nbytes/2**30:.2f} GiB pixels')
    missing=[]; reused=0
    for job in jobs:
        rec=job[3]
        key=(str(Path(rec['dir']).resolve()),RULES['order'],tuple(sorted(rec['files'])))
        saved=_RSNA_ORDER_MEMO.get(key)
        if saved is not None:
            rec['ordered']=list(saved[0]);reused+=1
        else: missing.append((job,key))
    def ordered(item):
        job,key=item
        try:
            files,good=order_slices(job[3])
            if len(files)!=len(job[3]['files']) or set(files)!=set(job[3]['files']):
                raise ValueError(f'ordering lost/added files ({len(files)} vs {len(job[3]["files"])})')
        except Exception as exc:
            rsna_event('ordering_fallback',tag=tag,series=str(job[3].get('SeriesInstanceUID')),error=f'{type(exc).__name__}: {exc}')
            return key,sorted(job[3]['files']),False
        return key,files,bool(good)
    with ThreadPoolExecutor(max_workers=ORDER_THREADS) as pool:
        for start in range(0,len(missing),256):
            rsna_deadline(tag+' ordering')
            block=missing[start:start+256]
            for (job,_),(key,files,good) in zip(block,pool.map(ordered,block)):
                job[3]['ordered']=files;_RSNA_ORDER_MEMO[key]=(tuple(files),good)
                if not good: rsna_event('deterministic_order_fallback',series=str(job[3].get('SeriesInstanceUID')))
    done=0; unfilled=0
    def _tolerant_read_slot(j):
        try: return read_slot(j[3],CACHE_SLICES,IMG)
        except Exception as exc:
            rsna_event('slot_read_error',tag=tag,study=str(j[0]),series=str(j[3].get('SeriesInstanceUID')),error=f'{type(exc).__name__}: {exc}'); return None
    with ThreadPoolExecutor(max_workers=PIX_THREADS) as pool:
        for start in range(0,len(jobs),128):
            rsna_deadline(tag+' decoding')
            block=jobs[start:start+128]
            for (st,k,plane,rec),img in zip(block,pool.map(_tolerant_read_slot,block)):
                done+=1
                if img is None:
                    unfilled+=1; rsna_event('slot_unfilled',tag=tag,study=str(st),series=str(rec.get('SeriesInstanceUID')),slot=int(k)); continue
                cache[sidx[st],k]=normalise_laterality(img,plane,lat_map.get(st)).numpy()
                mask[sidx[st],k]=1
    if unfilled: log(f'{tag}: {unfilled} slot(s) left empty and masked (flagged)')
    _empty=[studies[i] for i in np.flatnonzero(mask.sum(1)==0)]
    if _empty:
        rsna_event('empty_study_rows',tag=tag,count=len(_empty),studies=[str(s) for s in _empty])
        log(f'{tag}: {len(_empty)} study(ies) with no series matching this layout kept as all-zero masked rows (parent policy): {_empty[:5]}')
    if done!=len(jobs):
        raise RuntimeError(f'{tag}: incomplete cache; {done}/{len(jobs)} slots')
    if isinstance(cache,np.memmap): cache.flush()
    rsna_event('cache_complete',tag=tag,studies=len(studies),slot_jobs=len(jobs),filled=done,order_cache_hits=reused,bytes=int(cache.nbytes))
    log(f'{tag}: COMPLETE {done}/{len(jobs)}; reused ordering for {reused} series')
    return studies,cache,mask