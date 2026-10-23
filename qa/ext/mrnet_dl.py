"""Download MRNet v1.0 from Redivis (aimi.mrnet_knee_mri_s:4a2c:v1_0) into <out>/{train,valid}/<plane>/NNNN.npy, the
layout qa/ext/convert_ext.py reads. Resumable: a file whose md5 matches the index is skipped.
    REDIVIS_API_TOKEN=<token> python qa/ext/mrnet_dl.py <out>"""
import base64
import hashlib
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import redivis

out = Path(sys.argv[1])
ds = redivis.organization("aimi").dataset("mrnet_knee_mri_s:4a2c:v1_0")


def md5_b64(p):
    return base64.b64encode(hashlib.md5(p.read_bytes()).digest()).decode() if p.exists() else None


def fetch(row):
    rel = Path(row.file_name)
    if len(rel.parts) != 2 or rel.parts[0] not in ("axial", "coronal", "sagittal") or rel.suffix != ".npy":
        raise ValueError(f"unexpected file name {row.file_name}")
    dst = out / row.split / rel
    if md5_b64(dst) == row.md5_hash:
        return "skip"
    dst.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(3):
        tables[row.split].file(row.file_name).download(str(dst), overwrite=True, progress=False)
        if md5_b64(dst) == row.md5_hash:
            return "ok"
    raise IOError(f"md5 mismatch after 3 tries: {row.file_name}")


rows, tables = [], {}
for split, ref in (("train", "train:5wjf"), ("valid", "valid:zcmk")):
    tables[split] = ds.table(ref)
    df = tables[split].to_pandas_dataframe(progress=False)
    df["split"] = split
    print(split, len(df), "files", round(df["size"].sum() / 2**30, 2), "GiB", flush=True)
    rows += list(df.itertuples(index=False))
with ThreadPoolExecutor(16) as ex:
    res = list(ex.map(fetch, rows))
print({r: res.count(r) for r in set(res)})
for split in ("train", "valid"):
    print(split, {p: len(list((out / split / p).glob("*.npy"))) for p in ("axial", "coronal", "sagittal")})
print("MRNET_DL_DONE")
