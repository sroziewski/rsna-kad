"""bprefit gold-58 screen (information only: 58 studies, LB-gold slope ~0.24). Both arms are bpblur's blend on the same
inputs: MTR / MTH / OLP = the blurred (0.65 after the warp) all-fold gold predictions from qoof_driver.py, r7 blur, v5c,
hx48 = Andrey's v6 (as .tmp/research_gold/repro_b.py); only the stacker JSON differs (B's vs the refit).

  python -I gold_screen.py MEMBER_DIR NEW_STK.json
"""
import json
import sys
from pathlib import Path

mdir, new_path = sys.argv[1:3]
src = Path("/raid/sroziewski/sandbox/rsna_kad/.tmp/research_gold/repro_b.py").read_text()
exec(src[:src.index("qile_u = ")])
NEW = json.loads(Path(new_path).read_text())
for k, f in (("MTR", "member_mt_r50.csv"), ("MTH", "member_mt_hires.csv"), ("OLP", "member_ol_plane.csv")):
    g = pd.read_csv(Path(mdir) / f, dtype={ID_COL: str}).set_index(ID_COL)
    g = g[g.set == "gold58"]
    assert sorted(g.index) == ids, k
    MEM[k + "_b"] = g.loc[ids, cols].astype(float)


def blend(stk):
    M = dict(MTR=MEM["MTR_b"], MTH=MEM["MTH_b"], OLP=MEM["OLP_b"], v5c=MEM["v5c"], hx48=MEM["hx48_andrey"], r7=MEM["r7_blur"])
    RK = {k: {f: prk(v[f].values) for f in cols} for k, v in M.items()}
    X = np.stack([RK[KEY[f.split(" ", 1)[0]]][f.split(" ", 1)[1]] for f in stk["features"]], 1)
    X = (X - np.array(stk["mean"])) / np.array(stk["std"])
    out = np.zeros((len(ids), len(cols)))
    for j, f in enumerate(cols):
        qile = 0.1 * RK["MTR"][f] + 0.4 * RK["MTH"][f] + 0.5 * RK["OLP"][f]
        anchor = (0.40 * prk(qile) + 0.12 * RK["v5c"][f] + 0.20 * RK["hx48"][f]) / 0.72
        stack = 0.5 * prk(X @ np.array(stk["lr"][f])) + 0.5 * prk(X @ np.array(stk["ranklr"][f]))
        out[:, j] = 0.28 * RK["r7"][f] + 0.72 * prk(0.8 * anchor + 0.2 * stack)
    return out


qb = np.stack([prk(0.1 * prk(MEM["MTR_b"][f].values) + 0.4 * prk(MEM["MTH_b"][f].values) + 0.5 * prk(MEM["OLP_b"][f].values))
               for f in cols], 1)
ps = MEM["P_s0.65"].rank(pct=True).to_numpy()
print(f"sanity: blurred per-member Qile blend vs P_s0.65 leg max |rank d| {np.abs(qb - ps).max():.3f}, gold {m(qb):.4f} vs {m(ps):.4f}")
old, new = blend(STK), blend(NEW)
rng = np.random.default_rng(0)
boot = [rng.integers(0, len(ids), len(ids)) for _ in range(2000)]
d = np.array([m(new, ix) - m(old, ix) for ix in boot])
print(f"bpblur (B stacker) gold {m(old):.4f}  bprefit {m(new):.4f}  d {m(new) - m(old):+.4f}  90% [{np.percentile(d, 5):+.4f}, "
      f"{np.percentile(d, 95):+.4f}]  P>0 {np.mean(d > 0):.2f}")
print("per finding d:", ", ".join(f"{k} {b - a:+.3f}" for k, a, b in zip(cols, c.per_label(y, old), c.per_label(y, new))))
