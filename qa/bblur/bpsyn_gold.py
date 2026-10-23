"""Gold-58 screen of bpsyn vs bpblur (information only: 58 studies, LB-gold slope ~0.24). bpblur = B rebuilt with the
Qile part = P blur 0.65 (.tmp/research_gold/repro_b.py build, qile_override "P_s0.65"); the Synovitis stacker's
MTH / OLP inputs are the unblurred per-member gold files (no blurred per-member gold exists), hx48 = Andrey's v6.
"""
import ast
import json
from pathlib import Path

src = Path("/raid/sroziewski/sandbox/rsna_kad/.tmp/research_gold/repro_b.py").read_text()
exec(src[:src.index("qile_u = ")])
f4 = Path("/raid/sroziewski/sandbox/rsna_kad/.tmp/research_andrey/flat4synr6/rsna-knee-flat4-syn-r6.py").read_text()
SYN = next(json.loads(n.value.args[0].value) for n in ast.parse(f4).body
           if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "STK")
assert SYN["finding"] == "Synovitis" and len(SYN["features"]) == 36, SYN.get("note")
base = build("r7_blur", "hx48_andrey", qile_override="P_s0.65")[0]
M = dict(MEM, hx48=MEM["hx48_andrey"])
RK = {k: {f: prk(M[k][f].values) for f in cols} for k in ("MTH", "OLP", "hx48")}
KS = {"MT-hires": "MTH", "OL-plane": "OLP", "hx48": "hx48"}
XS = np.stack([RK[KS[f.split(" ", 1)[0]]][f.split(" ", 1)[1]] for f in SYN["features"]], 1)
XS = (XS - np.array(SYN["mean"])) / np.array(SYN["std"])
stack = 0.5 * prk(XS @ np.array(SYN["lr"])) + 0.5 * prk(XS @ np.array(SYN["ranklr"]))
j = cols.index(SYN["finding"])
new = base.copy()
new[:, j] = (1 - SYN["share"]) * prk(base[:, j]) + SYN["share"] * prk(stack)
rng = np.random.default_rng(0)
boot = [rng.integers(0, len(ids), len(ids)) for _ in range(2000)]
d = np.array([m(new, ix) - m(base, ix) for ix in boot])
ps = lambda a: c.per_label(y[:, [j]], (a[:, j] if a.ndim == 2 else a)[:, None])[0]
print(f"bpblur gold {m(base):.4f}  bpsyn {m(new):.4f}  d {m(new) - m(base):+.4f}  90% [{np.percentile(d, 5):+.4f}, "
      f"{np.percentile(d, 95):+.4f}]  P>0 {np.mean(d > 0):.2f}")
print(f"Synovitis AUC: bpblur {ps(base):.4f}  stacker alone {ps(stack):.4f}  bpsyn {ps(new):.4f}  "
      f"(positives {int(y[:, j].sum())}/58)")
