import json, glob, sys
root = sys.argv[1] if len(sys.argv) > 1 else "runs/main_convnext_s_384"
for p in sorted(glob.glob(root + "/fold*/metrics.json")):
    m = json.load(open(p))
    h = m["history"]
    print(p, "sel_ep", m["epoch"], "best_ep", m["best_epoch"]["epoch"])
    print("  weak   :", " ".join("%d:%.4f" % (x["epoch"], x["oof_weak_auc"]) for x in h))
    print("  gold   :", " ".join("%d:%.4f" % (x["epoch"], x["gold_auc"] if x["gold_auc"] is not None else float("nan")) for x in h))
    print("  strict :", " ".join("%d:%.4f" % (x["epoch"], x["oof_strict_auc"]) for x in h))
    print("  train_s:", " ".join("%.0f" % x["train_s"] for x in h),
          "| sum train %.0f eval %.0f total_h %.2f" % (sum(x["train_s"] for x in h), sum(x["eval_s"] for x in h),
                                                       (sum(x["train_s"] for x in h) + sum(x["eval_s"] for x in h)) / 3600.0))
