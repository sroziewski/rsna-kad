import numpy as np, pandas as pd, cv2, pydicom
from pathlib import Path
CACHE = Path("/raid/sroziewski/cache/rsna_kad/v1"); DATA = Path("/raid/sroziewski/data/rsna_kad")
OUT = Path("/raid/sroziewski/sandbox/rsna_kad/qa/stage1/png")
df = pd.read_parquet(CACHE / "index.parquet")
big = df[df.fov_mm >= 230]
print("series with fov>=230:", len(big), big.groupby(["manufacturer", "slot"]).size().to_dict())
print("selected among them:", int(big.selected.sum()), " main-slot selected:", int((big.selected & big.slot.isin(["sag_fs", "cor_fs", "ax_fs", "sag_t1"])).sum()))
print(big[["fov_mm", "rows", "spacing_mm", "mean_u8", "n_src"]].describe().round(1).to_string())
# montage of 4 big-FOV series: original middle (letterboxed) with crop box + cached middle
tiles = []
for r in big.sort_values("mean_u8").head(2).itertuples(index=False):
    pass
sel = pd.concat([big.sort_values("mean_u8").head(2), big[big.slot == "ax_fs"].head(1), big[big.slot == "sag_fs"].head(1)])
for r in sel.itertuples(index=False):
    d = DATA / f"{r.split}_series" / r.StudyInstanceUID / r.SeriesInstanceUID
    files = sorted(d.glob("*.dcm"))
    hdr = [pydicom.dcmread(f, stop_before_pixels=True) for f in files]
    files = [files[i] for i in np.argsort([int(h.InstanceNumber) for h in hdr])]
    ds = pydicom.dcmread(files[len(files) // 2]); img = ds.pixel_array.astype(np.float32)
    lo, hi = np.percentile(img, [1, 99]); u8 = np.clip((img - lo) * 255 / max(hi - lo, 1), 0, 255).astype(np.uint8)
    h, w = u8.shape; sc = 384 / max(h, w)
    o = cv2.resize(u8, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((384, 384), np.uint8); y, x = (384 - o.shape[0]) // 2, (384 - o.shape[1]) // 2
    canvas[y:y + o.shape[0], x:x + o.shape[1]] = o
    canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    s = int(round(130 / float(ds.PixelSpacing[0]))); y0, x0 = (h - s) // 2, (w - s) // 2
    cv2.rectangle(canvas, (int(x + x0 * sc), int(y + y0 * sc)), (int(x + (x0 + s) * sc), int(y + (y0 + s) * sc)), (0, 0, 255), 1)
    cv2.putText(canvas, f"{r.manufacturer[:10]} {r.slot} fov={r.fov_mm:.0f} {h}x{w}", (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    c = np.load(CACHE / r.split / r.StudyInstanceUID / f"{r.SeriesInstanceUID}.npy", mmap_mode="r")
    cm = cv2.cvtColor(np.asarray(c[c.shape[0] // 2]).copy(), cv2.COLOR_GRAY2BGR)
    cv2.putText(cm, f"cached mid mean={r.mean_u8:.0f} p={r.p_lo:.0f}/{r.p_hi:.0f}", (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    tiles.append(np.hstack([canvas, cm]))
cv2.imwrite(str(OUT / "13_bigfov.png"), np.vstack(tiles))
