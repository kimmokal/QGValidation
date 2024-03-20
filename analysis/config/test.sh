#!/usr/bin/env bash
OUT_ID="test"
FILES="${COFFEAHOME}/data/UL17/test_QCD_HT.txt"

python $COFFEAHOME/scripts/run_qg_analysis.py \
    --out_id $OUT_ID \
    --channel dijet \
    --files $FILES \
    --run mc \
    --workers 1 \
    --maxchunk 1
