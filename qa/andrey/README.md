# My CoAtNet ensemble (rsna-knee-native-hx48; public LB 0.948 alone, submission #32)

Two CoAtNet-RMLP-2 MIL models on my own native DICOM pipeline (DICOM -> in-memory export -> YOLOX knee landmarks on CPU -> typed series
slots, 160 mm wide + 96 mm knee-centred crops at 320 px -> every-slice inference), combined per finding on percentile ranks over the test
studies: Fracture / Lateral Meniscus / Lateral OA from model 2, Synovitis / Effusion from model 1, the other 7 findings = mean of both ranks.
- model 1 `coatnet_hx_swa.pt`: 12 training positions, metadata embeddings, canonical side (needs `side_tiny_5f.pt`), trained on
  competition data + MRNet + KneeMRI pseudo-labels. Alone: public 0.945.
- model 2 `coatnet_n48_swa.pt`: 48 training positions, competition data only.
Output values are per-finding percentile ranks in (0, 1] (rank-blend friendly).

## Run
- `hx48_ensemble.py`: the scored notebook's code cell as a script, unchanged except the working folder: `HX48_WORK=/kaggle/working/hx48_ensemble
  python hx48_ensemble.py` writes `$HX48_WORK/submission.csv` plus its own intermediates (`hx48_code/`, `hx48_pass1_{g}.npz`, `hx48_pass2_{g}.npz`, logs).
  Default HX48_WORK = /kaggle/working (= the scored notebook; it would overwrite /kaggle/working/submission.csv).
- `rsna-knee-native-hx48.ipynb` + `kernel-metadata.json`: kernel andreytikhomirov/rsna-knee-native-hx48, code cell = the scored notebook (submission #32) with the two checkpoint file names changed (files byte-identical, SHA-256 checked).
- All my Python sources are embedded in the script (FILES dict); nothing else to install.
- It uses every visible GPU (one subprocess per GPU, half of the test studies each), two passes in sequence (model 1, then model 2, which run
  with different flags). Run it while no other process holds the GPUs.

## Datasets to attach (both private, ~590 MB)
- `andreytikhomirov/rsna-knee-native-hx48-weights` (coatnet_hx_swa.pt, coatnet_n48_swa.pt, side_tiny_5f.pt)
- `andreytikhomirov/rsna-knee-landmark-detector` (YOLOX-nano knee landmarks + vendored YOLOX code, Apache-2.0)
- the competition data. Checkpoints are found by file name anywhere under /kaggle/input.

## Environment
Latest Kaggle Python image (no pinned Docker image), GPU T4 x2, internet off; uses the image's torch / timm.

## Time on the hidden test
Submit -> scored: model 1 alone 52 min; both models (this ensemble) 106 min, i.e. ~45-50 min per pass on 2xT4. The visible-test run takes ~70 s.

## External data
MRNet (Stanford AIMI, registration + research-use terms) and KneeMRI (Zenodo, open download, CC BY-NC-ND 4.0) as pseudo-labelled training
studies for model 1. No OAI, no fastMRI, no public competition weights. Backbone init: timm `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k`.

## Gold-58
Gold never in training. Leg (as scored) 0.936 macro AUC; last-epoch version 0.937.
