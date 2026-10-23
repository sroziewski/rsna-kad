#!/usr/bin/env bash
# r6base gold-58 e2e (2026-10-17): qa/v2stack/e2e.sh with goodpjw2008's 0.958 notebook (v27 = v25's recipe: nartaa
# Raptor 384 + 224, heliosli R6 with flip TTA, per-finding Raptor share, goodpjw's reader at 1/10) as the base of our
# V leg instead of the R7 notebook; main_v5_C at 0.3 as in v3stack. Then qa/r6base/gold_r6.py.
cd /raid/sroziewski/sandbox/rsna_kad
NB=qa/r6base/src/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958.ipynb
GPU=${GPU:-1} ONLY=full TAG=r6_ \
  EXTRA_DS="rostigor/rsna-knee-r6-fe055406=rostigor/rsna-knee-r6-fe055406 goodpjw2008/rsna-knee-2-5d-convnext-reader=goodpjw_reader" \
  BUILD_ARGS="--generic-nb --nb $NB --nb-sha a906df9328b6a53d7e0fe4e18962c655867c8617c54e6745a8af647033b2399c" \
  bash qa/v2stack/e2e.sh
echo R6_E2E_DONE
