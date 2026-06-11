#!/usr/bin/env bash
# Sequential video-diffusion training: 01 (axis-x) then 03 (sphere), ONE at a time
# (never concurrent). The stack-first picker piles onto an in-use GPU when memory
# fits (1 card, no penalty); the 4h watchdog + per-trainer quota_stop(4h20m) + resume
# are the budget safety. Re-runnable: --resume auto continues each from latest.pt.
set +e
GP=/tmp2/b10401006/.symlinks/miniforge3/envs/genphys-diff/bin/python
cd /tmp2/b10401006/ev-project/generative-phys/teacher

for D in 01_tel_axisx_rest_T16 03_tel_sphere_rest_T16; do
  echo "########## [$(date '+%F %T')] TRAIN $D ##########"
  "$GP" train_video.py --pack "../outputs/dataset_gen/$D/video_pack_128.npy" \
    --out "out_$D" --res 128 --frames 16 --epochs 100 \
    --sample_every 50 --ckpt_every 10 --quota_stop_secs 15600 --max_my_gpus 8 --resume auto
  echo "########## [$(date '+%F %T')] done $D (exit=$?) ##########"
done
echo "########## [$(date '+%F %T')] ALL TRAINING DONE ##########"
