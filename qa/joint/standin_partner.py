"""Stand-in for the partner's inference in the joint kernel's gold-58 e2e (qa/joint): reads Kaggle paths as a partner
notebook would, writes a relative submission.csv, from precomputed predictions of a model that is not in leg V
(main_v5_C_oai 6 seeds).  STANDIN_MODE = ok | fail | partial | sleep exercises the joint kernel's fallbacks."""
import os
import time

import pandas as pd

ID = "StudyInstanceUID"
mode = os.environ.get("STANDIN_MODE", "ok")
test = pd.read_csv('/kaggle/input/competitions/rsna-knee-abnormality-detection/test.csv', dtype={ID: str})
pred = pd.read_csv('/kaggle/input/datasets/standin/partner-standin/preds.csv', dtype={ID: str}).set_index(ID)
pred = pred.loc[test[ID]]
if mode == "fail":
    raise SystemExit(3)
if mode == "sleep":
    time.sleep(10 ** 6)
if mode == "partial":
    pred = pred.iloc[:-1]
for f in os.listdir("."):
    if f.endswith(".csv") and f != "submission.csv":
        os.remove(f)  # a partner may clean its working dir; leg V's saved copy must survive this
pred.reset_index().to_csv("submission.csv", index=False)
print("standin wrote", len(pred), "rows, mode", mode)
