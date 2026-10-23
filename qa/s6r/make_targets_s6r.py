"""s6r targets: our v3_pseudo_full labels in the format of nartaa's S6 trainer (their make_data.py, same files).

targets_s6f0.csv: StudyInstanceUID, is_gold, review_holdout (0), fold (folds_v0; gold -1), L|prior = our target,
L|strict blank, L|w = our cell weight (trained through the custom arm OURS=bce:prior:w). Gold-58 rows have blank labels and
are never trained (v3_pseudo_full has no gold rows). The other files are their make_data.py outputs with our pool: every
non-gold study of folds_v0 (all 4,349 have DICOMs here) + the 58 gold studies for the precompute.
"""
import csv
import io
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/raid/sroziewski/sandbox/rsna_kad")
R = ROOT / ".tmp/s6r"
DATA_DIR = R / "S6F0/data"
OURS = ROOT / "data/targets/v3_pseudo_full/full.parquet"
DICOM = Path("/raid/sroziewski/data/rsna_kad/train_series")
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
          "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
KEYS = ["acl", "mcl", "med_men", "lat_men", "med_oa", "lat_oa", "pf_oa", "effusion", "synovitis", "baker", "contusion",
        "fracture"]


def dump(obj, name):
    tmp = DATA_DIR / (name + ".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, DATA_DIR / name)


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    folds = json.load(open(R / "folds_v0.json"))["folds"]
    ours = pd.read_parquet(OURS).set_index("StudyInstanceUID")
    on_disk = set(os.listdir(DICOM))
    gold = sorted(u for u, f in folds.items() if f == -1)
    tr = pd.read_csv(Path("/raid/sroziewski/data/rsna_kad/train.csv"))
    gold_csv = set(tr.loc[tr[LABELS].notna().all(axis=1), "StudyInstanceUID"].astype(str))
    assert set(gold) == gold_csv and len(gold) == 58 and not (set(gold) & set(ours.index))
    non_gold = sorted(u for u, f in folds.items() if f >= 0)
    assert set(non_gold) == set(ours.index), "folds_v0 non-gold != our label rows"
    assert set(non_gold + gold) <= on_disk, "a study has no DICOM folder"
    t = ours[[k + "_target" for k in KEYS]].to_numpy(np.float64)
    w = ours[[k + "_weight" for k in KEYS]].to_numpy(np.float64)
    assert np.isfinite(t).all() and np.isfinite(w).all() and ((t >= 0) & (t <= 1)).all() and (w >= 0).all()
    buf = io.StringIO()
    wr = csv.writer(buf, lineterminator="\n")
    wr.writerow(["StudyInstanceUID", "is_gold", "review_holdout", "fold"] + [L + "|prior" for L in LABELS]
                + [L + "|strict" for L in LABELS] + [L + "|w" for L in LABELS])
    for i, u in enumerate(ours.index):
        wr.writerow([u, 0, 0, int(folds[u])] + [repr(float(x)) for x in t[i]] + [""] * len(LABELS)
                    + [repr(float(x)) for x in w[i]])
    for u in gold:
        wr.writerow([u, 1, 0, -1] + [""] * (3 * len(LABELS)))
    (DATA_DIR / "targets_s6f0.csv").write_text(buf.getvalue())
    train = [u for u in non_gold if folds[u] in (1, 2, 3, 4)]
    f0 = [u for u in non_gold if folds[u] == 0]
    rng = np.random.default_rng(42)
    smoke = sorted(rng.choice(train, 24, replace=False).tolist())
    dump(train, "train_ids.json")
    dump(f0, "fold0_ondisk.json")
    dump(f0, "fold0_all.json")
    dump(gold, "gold58.json")
    dump(smoke, "smoke_train.json")
    dump(f0[:4] + gold[:2], "smoke_ho.json")
    dump(smoke + [u for u in train + f0 + gold if u not in set(smoke)], "precompute_order.json")
    print(f"targets: {len(ours)} labelled + {len(gold)} gold rows; train folds 1-4 {len(train)}, fold 0 {len(f0)}, "
          f"precompute {len(train) + len(f0) + len(gold)}; mean weight {w.mean():.3f}, mean target {t.mean():.3f}")


if __name__ == "__main__":
    main()
