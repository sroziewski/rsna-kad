#!/usr/bin/env bash
# 2026-10-18 scan: crawl8 (= crawl7 with 10-15 seeds + new_since_1015.txt) in screen, competition kernel listings now.
cd /raid/sroziewski/sandbox/rsna_kad
N=qa/discussion_2026-10-18
sed -e "s#2026-10-15 refresh (crawl6.py with 10-14 kept topics as seeds, merge/final terms added)#2026-10-18 refresh (crawl7.py, 10-15 seeds)#" \
    -e "s#qa/discussion_2026-10-15#$N#" \
    -e "s#qa/discussion_2026-10-14/competition_topics.json#qa/discussion_2026-10-15/competition_topics.json#" \
    qa/discussion_2026-10-15/crawl7.py > $N/crawl8.py
cat $N/crawl8_tail.py >> $N/crawl8.py
screen -dmS crawl8 bash -c "/raid/sroziewski/sandbox/bin/python -I $N/crawl8.py > $N/crawl8.log 2>&1; echo CRAWL_DONE >> $N/crawl8.log"
for s in dateRun scoreDescending voteCount; do
  bash scripts/kaggle_cli.sh raw kernels list --competition rsna-knee-abnormality-detection --sort-by $s --page-size 40 > $N/kernels_$s.txt 2>&1
done
