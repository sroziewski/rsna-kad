# Getting the competition data onto the GPU server

Moved here from the old top-level README when the repository was published.

Target: `<user>@<gpu-server>:/raid/<user>/data/rsna_kad`.
Everything the fetcher needs lives in `/raid/sroziewski/sandbox/rsna_kad`
(script, token, tests); it runs with `/raid/sroziewski/sandbox/bin/python`,
needs no pip packages and never writes to the home directory.

One-time, in a browser:

1. Accept the rules: <https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules>
2. Create a token: <https://www.kaggle.com/settings/api> -> "Generate New Token"

Then from this machine (PowerShell 7):

```powershell
.\scripts\remote.ps1 deploy      # copy the fetcher to the server
.\scripts\remote.ps1 token       # paste the token; stored on the server with mode 600
.\scripts\remote.ps1 preflight   # token OK? rules accepted? archive size? disk space?
.\scripts\remote.ps1 start       # full fetch in a detached screen session "fetcher"
.\scripts\remote.ps1 status      # running? last log lines, disk usage
.\scripts\remote.ps1 logs        # follow the log (Ctrl+C leaves; the fetch keeps running)
.\scripts\remote.ps1 stop        # interrupt; "start" resumes where it stopped
```

On the server: `screen -r fetcher` to watch, `Ctrl+A, D` to detach.
A browser session cannot be reused for this - Kaggle's API only accepts API
tokens, and scraping session cookies breaks on every login/expiry.

### Why not just `kaggle competitions download`?

The dataset is ~570 GB in ~820k files and `/raid` has ~960 GB free. The
official CLI downloads one zip and unpacks it next to itself, which needs
zip + extracted data on disk together (> 1 TB) and has no resume. The fetcher:

- downloads with HTTP range requests - any interruption resumes from the last
  byte, expired signed URLs are renewed, a republished archive is detected;
- extracts in archive order with 8 threads and, when space is short
  (`--reclaim auto`), hands the already-extracted part of the zip back to the
  filesystem (`fallocate` hole punching) so peak usage stays near the size of
  the dataset instead of twice that;
- checks every file's CRC32 while extracting, writes
  `.download/manifest.tsv` (crc, size, path) and verifies the tree at the end;
- keeps a reserve (`--min-free-gib`, default 20) because the disk is shared.

In reclaim mode the zip is consumed; if an extracted file is lost afterwards
the only fix is downloading again.

Fallback if Kaggle changes its API routes: install the official CLI into the
sandbox (`/raid/sroziewski/sandbox/bin/pip install --no-cache-dir kaggle`), download
with `KAGGLE_CONFIG_DIR=/raid/sroziewski/sandbox/rsna_kad/.kaggle kaggle competitions download
rsna-knee-abnormality-detection -p /raid/sroziewski/data/rsna_kad/.download`, then run
`kaggle_fetch.py extract` - it picks the zip up from there.

## How this was built (role plan)

| # | Role | Output |
|---|------|--------|
| 1 | Researcher | Kaggle auth methods (token / legacy json), the two live API routes that issue the archive redirect, CLI limitations (no resume, unzip-in-place), dataset size |
| 2 | Analyst | Server facts: 961 GB free on shared XFS, Python 3.12 venv without `kaggle`/`requests`, `screen` available, no credentials anywhere -> zip + data cannot coexist; token is the one manual step |
| 3 | Architect | Stdlib-only single script, four resumable phases, reclaim-while-extracting with a sync -> record -> punch ordering, state in `<data>/.download`, thin PowerShell launcher + `screen` |
| 4 | Developer | `scripts/kaggle_fetch.py`, `scripts/remote.ps1` |
| 5 | Tester | `tests/test_kaggle_fetch.py`: fake Kaggle API + range server (drops, expiry, republish), synthetic zips, reclaim/resume/corruption on the real filesystem; run with `remote.ps1 test` |
| 6 | Reviewer | Independent review focused on data-loss windows, resume logic, credential leakage |

Not testable without your token: the authenticated calls against the real
Kaggle API. `preflight` exercises exactly those and nothing else, so run it first.
