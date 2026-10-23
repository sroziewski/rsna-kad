"""Where is the anatomy in the cached frame? Intensity centroid offset and border brightness
of the middle slice of every series (read-only, 6 processes)."""
import multiprocessing as mp
from pathlib import Path
import numpy as np, pandas as pd

CACHE = Path("/raid/sroziewski/cache/rsna_kad/v1")
df = pd.read_parquet(CACHE / "index.parquet")
YY, XX = np.mgrid[0:384, 0:384]


def one(args):
    split, study, series = args
    a = np.load(CACHE / split / study / f"{series}.npy", mmap_mode="r")
    m = np.asarray(a[a.shape[0] // 2]).astype(np.float32)
    w = m.sum()
    cy, cx = (YY * m).sum() / w - 191.5, (XX * m).sum() / w - 191.5
    border = np.concatenate([m[:24].ravel(), m[-24:].ravel(), m[:, :24].ravel(), m[:, -24:].ravel()]).mean()
    centre = m[96:288, 96:288].mean()
    return cy, cx, border, centre


if __name__ == "__main__":
    args = list(zip(df.split, df.StudyInstanceUID, df.SeriesInstanceUID))
    with mp.get_context("forkserver").Pool(6) as pool:
        res = pool.map(one, args, chunksize=64)
    r = pd.DataFrame(res, columns=["cy", "cx", "border", "centre"])
    r["off"] = np.hypot(r.cy, r.cx)
    r["ratio"] = r.border / (r.centre + 1e-3)
    out = pd.concat([df[["SeriesInstanceUID", "manufacturer", "slot", "selected", "fov_mm", "mean_u8"]].reset_index(drop=True), r], axis=1)
    out.to_csv(Path(__file__).with_name("centroid.csv"), index=False)
    print("centroid offset px (mm = px*0.339): quantiles 50/90/99/max:", np.percentile(out.off, [50, 90, 99]).round(1).tolist(), round(out.off.max(), 1))
    print("offset > 60 px (20 mm):", int((out.off > 60).sum()), " > 90 px (30 mm):", int((out.off > 90).sum()))
    print("by slot, offset>60:", out[out.off > 60].groupby("slot").size().to_dict())
    print("by slot main selected, offset>60:", out[(out.off > 60) & out.selected].groupby("slot").size().to_dict())
    print("border/centre ratio quantiles 50/90/99:", np.percentile(out.ratio, [50, 90, 99]).round(2).tolist())
    print("ratio > 1 (border brighter than centre):", int((out.ratio > 1).sum()), out[out.ratio > 1].groupby("slot").size().to_dict())
    print("centre mean < 15 (empty centre):", int((out.centre < 15).sum()), out[out.centre < 15].groupby(["manufacturer", "slot"]).size().to_dict())
    print(out.sort_values("off", ascending=False).head(15)[["manufacturer", "slot", "selected", "fov_mm", "mean_u8", "cy", "cx", "off", "ratio"]].round(1).to_string())
    print("fov>=230:", out[out.fov_mm >= 230].off.describe().round(1).to_dict())
