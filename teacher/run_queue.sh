#!/usr/bin/env bash
# Sequential GPU queue (one job at a time):
#   1) RGB-200 baseline dynamic hit-rate (24 samples, from the epoch-199 diff_final)
#   2) flow-200  (full flow pack, fresh)
#   3) RGB 200->400 (resume the RGB checkpoint)
#   4) RGB-400 dynamic hit-rate (24 samples)
# Steps don't abort each other (no set -e); each logs a marker.
cd "$(dirname "$0")"
PY=.venv/bin/python
D=../outputs/dataset_gen/01_tel_axisx_rest_T16
RGB=out_01_tel_axisx_T8_local/checkpoints/diff_final.pt
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mark(){ echo; echo "########## $* ($(date +%H:%M)) ##########"; }

mark "STEP1 RGB-200 baseline hit-rate"
$PY measure_hitrate.py --ckpt "$RGB" --n 24 --thresh 5 --label RGB200_baseline

mark "STEP2 flow-200 (full pack)"
$PY train_flow.py --pack $D/flow_pack_128_t8.npy --out out_01_flow_T8_local \
  --res 128 --dim 64 --dim_mults 1 2 4 8 --epochs 200 --batch 1 --grad_accum 4 \
  --sample_every 50 --ckpt_every 10 --resume auto

mark "STEP3 RGB 200->400 (resume)"
EPOCHS=400 ./train_local.sh

mark "STEP4 RGB-400 hit-rate"
$PY measure_hitrate.py --ckpt "$RGB" --n 24 --thresh 5 --label RGB400

mark "QUEUE DONE"
