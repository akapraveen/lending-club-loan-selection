#!/usr/bin/env bash
# Needs a Kaggle API token in ~/.kaggle/kaggle.json (kaggle.com > Settings > API > Create New Token).
set -euo pipefail
cd "$(dirname "$0")" && mkdir -p data
for f in accepted_2007_to_2018Q4.csv.gz rejected_2007_to_2018Q4.csv.gz; do
  kaggle datasets download -d wordsforthewise/lending-club -f "$f" -p data
  if [ -f "data/$f.zip" ]; then (cd data && unzip -o "$f.zip" && rm "$f.zip"); fi
done
ls -la data
