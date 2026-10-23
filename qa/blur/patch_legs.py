"""Inference blur for the joint legs P (Qile's notebook) and A (Andrey's hx48 v6), as patched copies (2026-10-19,
Andrey's finding: Gaussian sigma 0.5-0.8 on 2D slices before normalisation helps hx48 / MT-hires / Qile's blend on gold).

The originals are never edited. Each leg embeds the code that runs as a Python string literal (hx48: FILES["infer_fast.py"];
Qile: the ours_infer.py text cell 0 writes); that literal is located with ast and replaced in place by repr() of the patched
string, so every other byte of the file stays identical. The blur runs on the model-input grid, after the resize and before
the model's normalisation, every finding; SIGMA 0 makes it the identity (parity test).
  P: cv2.GaussianBlur per uint8 slice of decode()'s crops, rounded back to uint8 (the dtype study_input expects).
  A: separable Gaussian on the GPU over each (H, W) plane of the uint8 window batch, kept float (gpu_intensity takes float);
     kernel radius round(4 sigma) = the size cv2 picks for P, so both legs blur alike.

  python qa/blur/patch_legs.py --sigma 0.65 --qile-nb NB --hx48 PY --out DIR   # writes DIR/qile_blur.ipynb, DIR/hx48_blur.py
"""
import argparse
import ast
import json
from pathlib import Path

QILE_ANCHOR = ("        crops[(img, crop_mm)] = np.stack([cv2.warpAffine(s, M, (img, img), flags=cv2.INTER_LINEAR, "
               "borderValue=0) for s in src])   # uint8 [n, img, img]\n")
QILE_IMPORT = "import numpy as np, pandas as pd, cv2, torch, torch.nn as nn, torch.nn.functional as F, timm, timm.data\n"
QILE_FN = '''K1_BLUR_SIGMA = {sigma!r}   # [blur probe] Gaussian sigma in model-input pixels; 0 = off


def _k1_blur(a):
    if K1_BLUR_SIGMA <= 0:
        return a
    out = np.empty_like(a)
    for i in range(a.shape[0]):
        y = cv2.GaussianBlur(a[i].astype(np.float32), (0, 0), sigmaX=K1_BLUR_SIGMA, borderType=cv2.BORDER_REFLECT_101)
        out[i] = np.clip(np.rint(y), 0, 255).astype(np.uint8)
    return out
'''
HX_ANCHOR = 'x = gpu_intensity(bb["x"].to(dev, non_blocking=True), bb["ip"].to(dev), cfg.norm)'
HX_NEW = 'x = gpu_intensity(_k1_blur(bb["x"].to(dev, non_blocking=True)), bb["ip"].to(dev), cfg.norm)'
HX_IMPORT = "from train_native import Net, T\n"
HX_FN = '''K1_BLUR_SIGMA = {sigma!r}   # [blur probe] Gaussian sigma in model-input pixels; 0 = off


def _k1_blur(x):
    if K1_BLUR_SIGMA <= 0:
        return x
    r = max(1, (int(round(8 * K1_BLUR_SIGMA + 1)) | 1) // 2)   # = cv2.GaussianBlur's own float kernel size, as leg P uses
    g = torch.exp(-torch.arange(-r, r + 1, device=x.device, dtype=torch.float32) ** 2 / (2 * K1_BLUR_SIGMA ** 2))
    g = g / g.sum()
    shp = x.shape
    y = x.float().reshape(-1, 1, shp[-2], shp[-1])
    y = torch.nn.functional.conv2d(torch.nn.functional.pad(y, (r, r, 0, 0), mode="reflect"), g.view(1, 1, 1, -1))
    y = torch.nn.functional.conv2d(torch.nn.functional.pad(y, (0, 0, r, r), mode="reflect"), g.view(1, 1, -1, 1))
    return y.reshape(shp)
'''


def replace_literal(text, pick, patch):
    """Replace the one string literal of `text` chosen by pick(node) with repr(patch(value)); positions are utf-8 byte
    offsets (ast), so the work is done on bytes."""
    # notebook-only lines (! / %) are masked to a same-length comment so the parse positions stay valid for `text`
    masked = "".join("#" + ln[1:] if ln.lstrip()[:1] in ("!", "%") and ln[:1] in ("!", "%") else ln
                     for ln in text.splitlines(keepends=True))
    nodes = [n for n in ast.walk(ast.parse(masked)) if isinstance(n, ast.Constant) and isinstance(n.value, str) and pick(n)]
    assert len(nodes) == 1, f"{len(nodes)} literals matched"
    n = nodes[0]
    lines = text.encode("utf-8").splitlines(keepends=True)
    start = sum(len(x) for x in lines[:n.lineno - 1]) + n.col_offset
    end = sum(len(x) for x in lines[:n.end_lineno - 1]) + n.end_col_offset
    raw = text.encode("utf-8")
    # the value comes from the unmasked bytes: the mask also rewrites ! / % lines inside a triple-quoted literal
    value = ast.literal_eval(raw[start:end].decode("utf-8"))
    patched = patch(value)
    assert ast.literal_eval(repr(patched)) == patched
    return (raw[:start] + repr(patched).encode("utf-8") + raw[end:]).decode("utf-8")


def once(s, old, new):
    assert s.count(old) == 1, f"anchor found {s.count(old)} times: {old[:80]!r}"
    return s.replace(old, new)


def patch_qile_src(s, sigma):
    # every model family's input comes from this one crops assignment (decode); a second builder would stay unblurred
    assert s.count("crops[(img, crop_mm)] =") == 1 and s.count("cv2.warpAffine(") == 1, "Qile: crop builder not unique"
    s = once(s, QILE_IMPORT, QILE_IMPORT + "\n" + QILE_FN.format(sigma=float(sigma)) + "\n")
    return once(s, QILE_ANCHOR, QILE_ANCHOR + "        crops[(img, crop_mm)] = _k1_blur(crops[(img, crop_mm)])   # [blur probe]\n")


def patch_hx_src(s, sigma):
    # the one model-input path of infer_fast (both model sets share x); it runs outside the fp16 autocast block
    assert s.count("gpu_intensity(") == 1, "hx48: gpu_intensity call sites changed"
    s = once(s, HX_IMPORT, HX_IMPORT + "\n" + HX_FN.format(sigma=float(sigma)) + "\n")
    return once(s, HX_ANCHOR, HX_NEW)


def patch_qile_nb(nb_text, sigma):
    nb = json.loads(nb_text)
    hits = 0
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        try:
            new = replace_literal(src, lambda n: QILE_ANCHOR in n.value, lambda v: patch_qile_src(v, sigma))
        except AssertionError as e:
            if str(e) == "0 literals matched":
                continue
            raise
        c["source"] = new.splitlines(keepends=True)
        hits += 1
    assert hits == 1, f"{hits} notebook cells patched"
    return json.dumps(nb, indent=1, ensure_ascii=False) + "\n"


def files_literal(text, name):
    """pick() for the FILES[name] value literal of hx48_ensemble*.py."""
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "FILES":
            v = dict(zip([k.value for k in node.value.keys], node.value.values))[name]
            return lambda n: (n.lineno, n.col_offset) == (v.lineno, v.col_offset)
    raise AssertionError("FILES not found")


def patch_hx48(text, sigma):
    return replace_literal(text, files_literal(text, "infer_fast.py"), lambda v: patch_hx_src(v, sigma))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma", type=float, required=True)
    ap.add_argument("--qile-nb", required=True)
    ap.add_argument("--hx48", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "qile_blur.ipynb").write_text(patch_qile_nb(Path(a.qile_nb).read_text(encoding="utf-8"), a.sigma), encoding="utf-8")
    (out / "hx48_blur.py").write_text(patch_hx48(Path(a.hx48).read_text(encoding="utf-8"), a.sigma), encoding="utf-8")
    print(f"sigma {a.sigma}: wrote {out / 'qile_blur.ipynb'}, {out / 'hx48_blur.py'}")


if __name__ == "__main__":
    main()
