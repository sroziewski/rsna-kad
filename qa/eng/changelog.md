# Engineering changelog (developer P, audit 2026-09-25 P0/P1)

- 2026-09-26 item 1 VCS: server tree under git (`main`, baseline f789876, tag `main_v3-shipped` = kernel v10 notebook md5 e61dc63c); `.gitignore` keeps out token, caches, runs/export/logs, large tables.
- 2026-09-26 item 1 provenance: `export_kaggle.py` writes the tree's commit (`-dirty` if tracked files changed) into `ensemble.json` and every member json with an empty training `git_sha`; `run_screen.sh` exports `RSNA_KAD_GIT_SHA` so new training bundles carry it too.
- 2026-09-26 item 1 drift: `remote.ps1 sync` (and new `drift`) refuses when a differing server file is newer than the local copy or changed since the last sync (`.sync-manifest.tsv`); `-Force` overrides.
- 2026-09-26 item 2 export guard: `export_kaggle.py` refuses live runs, folds with `last.pt` or not at the configured final epoch, missing folds (was: silent skip + reweight) and `*_sweep_*`/`*smoke*`/`--limit` runs unless `--allow-smoke` (`src/rsna_kad/runguard.py`, tests/test_runguard.py).
- 2026-09-26 item 2 liveness: `predict_oof.py`, `evaluate_runs.py`, `compare_ensemble.py` refuse a run a live process still names; the caller's own ancestors are exempt, so a chain can still assemble its own OOF.
