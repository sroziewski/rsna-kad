# ========================= DIAGNOSTICS =====================================
# Two questions the leaderboard cannot answer, both cheap here because the two
# arms are still in memory:
#
#   agreement - if the transformer stack and the CoAtNet arm rank studies
#               almost identically for a finding, then the outer weight for
#               that finding cannot matter, whatever the probes suggested;
#   ties      - a block of identical scores earns half credit against every
#               study it ties with.
import numpy as _dg_np
import pandas as _dg_pd

try:
    _dg_labels = list(_blend_labels)
    _dg_rows = []
    for _dg_label in _dg_labels:
        left = _blend_tr[_dg_label].to_numpy(_dg_np.float64)
        right = _blend_cr[_dg_label].to_numpy(_dg_np.float64)
        final = _blend_output[_dg_label].to_numpy(_dg_np.float64)
        _dg_rows.append({
            "finding": _dg_label,
            "outer CoAtNet w": coat_w(_dg_label),
            "arm agreement": float(_dg_np.corrcoef(left, right)[0, 1]),
            "largest tie block": int(_dg_np.unique(final, return_counts=True)[1].max()),
        })
    _DG = _dg_pd.DataFrame(_dg_rows).set_index("finding")
    print(_DG.to_string(float_format=lambda v: f"{v:.3f}"))

    _dg_flat = _DG[_DG["arm agreement"] > 0.99]
    if len(_dg_flat):
        print("\nThese findings have two arms that already agree above 0.99, so "
              "their outer weight is close to a no-op no matter what it is set to:",
              ", ".join(_dg_flat.index), flush=True)
    _dg_loud = _DG[_DG["arm agreement"] < 0.85]
    if len(_dg_loud):
        print("\nThese findings have genuinely different arms, which is where an "
              "outer weight actually decides the ordering — and therefore where a "
              "weight fitted to the public split can do real damage privately:",
              ", ".join(_dg_loud.index), flush=True)
except NameError:
    print("[diag] fusion variables not in memory; run the final stage first", flush=True)