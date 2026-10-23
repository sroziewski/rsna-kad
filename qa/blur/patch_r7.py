"""Inference blur for R7 (heliosli's compact R7 inside our V leg), as a patched copy of the v3stack base notebook
(2026-10-19, Andrey's sweep: R7 +0.0101 gold-58 macro at sigma 0.288 mm on the raw DICOM pixels, P(gain) 99.8 %).

The notebook's cell 2 holds R7's sources as SOURCES = {name: text} and their pinned SOURCE_SHA = {name: sha256}.
probe_infer.install_native (R7's decoder hook, called by every R7 worker) gains one call that wraps
training.dicom_geometry.pixels_from_dataset: after the modality LUT / MONOCHROME1 step and before the series quantisation
and windowing, each slice is blurred with sigma = SIGMA_MM / PixelSpacing per axis (row spacing for y, column spacing for
x). Raptor and our C decode in their own processes with their own readers, so only R7 sees the blur (the side classifier
and knee anchors of R7 read the same decoded series, as in Andrey's measurement). SOURCES["probe_infer.py"] and
SOURCE_SHA["probe_infer.py"] are replaced in place (patch_legs.replace_literal); every other byte stays. SIGMA_MM 0 = off.

  python qa/blur/patch_r7.py --sigma-mm 0.288 --nb NB --out OUT.ipynb      # prints the new notebook sha256
"""
import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from patch_legs import once, replace_literal  # noqa: E402

ANCHOR_TOP = 'UIDK = "StudyInstanceUID"\n'
ANCHOR_CALL = "    import training.native_cache as native\n"
FN = '''K1_BLUR_MM = {sigma!r}   # [blur probe] Gaussian sigma in mm on the raw DICOM pixels; 0 = off


def _k1_blur_install(native):
    """Wrap pixels_from_dataset once per process: blur each decoded slice at K1_BLUR_MM / PixelSpacing per axis."""
    geom = native.geom
    if K1_BLUR_MM <= 0 or getattr(geom, "_k1_blur", False):
        return
    import dataclasses
    import cv2
    orig = geom.pixels_from_dataset

    def blurred(ds, image=None, path="archive"):
        p = orig(ds, image=image, path=path)
        sy, sx = K1_BLUR_MM / p.spacing[0], K1_BLUR_MM / p.spacing[1]
        img = np.asarray(p.image, np.float32)
        frames = img.reshape(-1, *img.shape[-2:])
        out = np.stack([cv2.GaussianBlur(f, (0, 0), sigmaX=sx, sigmaY=sy, borderType=cv2.BORDER_REFLECT_101)
                        for f in frames]).reshape(img.shape)
        return dataclasses.replace(p, image=out)

    geom.pixels_from_dataset = blurred
    geom._k1_blur = True

'''


def patch_probe_infer(s, sigma_mm):
    assert s.count("pixels_from_dataset") == 2, "probe_infer: decoder hook changed"
    s = once(s, ANCHOR_TOP, ANCHOR_TOP + "\n" + FN.format(sigma=float(sigma_mm)))
    return once(s, ANCHOR_CALL, ANCHOR_CALL + "    _k1_blur_install(native)   # [blur probe]\n")


def dict_value_pick(text, var, key):
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == var:
            v = dict(zip([k.value for k in node.value.keys], node.value.values))[key]
            return lambda n: (n.lineno, n.col_offset) == (v.lineno, v.col_offset)
    raise AssertionError(f"{var} not found")


def patch_cell(cell, sigma_mm):
    new_src = {}

    def p(v):
        new_src["text"] = patch_probe_infer(v, sigma_mm)
        return new_src["text"]
    cell = replace_literal(cell, dict_value_pick(cell, "SOURCES", "probe_infer.py"), p)
    sha = hashlib.sha256(new_src["text"].encode("utf-8")).hexdigest()   # the notebook writes text.encode('utf-8')
    return replace_literal(cell, dict_value_pick(cell, "SOURCE_SHA", "probe_infer.py"), lambda v: sha), sha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma-mm", type=float, required=True)
    ap.add_argument("--nb", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    assert a.sigma_mm >= 0, "sigma must be >= 0 (0 = off)"
    nb = json.loads(Path(a.nb).read_text(encoding="utf-8"))
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    hits = [c for c in code if "SOURCE_SHA = {" in "".join(c["source"]) and "SOURCES = {" in "".join(c["source"])]
    assert len(hits) == 1, len(hits)
    cell = "".join(hits[0]["source"])
    new, sha = patch_cell(cell, a.sigma_mm)
    hits[0]["source"] = new.splitlines(keepends=True)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"sigma {a.sigma_mm} mm: probe_infer.py sha256 {sha}; notebook {out} sha256 "
          f"{hashlib.sha256(out.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
