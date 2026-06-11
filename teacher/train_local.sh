#!/usr/bin/env bash
# Train the teacher video-diffusion LOCALLY (16GB 4090 Laptop) on the 01 T16
# axisx dataset, using a fits-on-this-card config.
#
# Memory fit (measured): res=128 x frames=8 peaks ~8.3GB (fp32, full 35.71M params).
# We take the FIRST 8 frames of the T16 pack (NOT an arbitrary window): v0 is
# physically meaningful, so the clip must start at t0 or the learned v0
# distribution drifts.
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
PACK_DIR="../outputs/dataset_gen/01_tel_axisx_rest_T16"
SRC_PACK="$PACK_DIR/video_pack_128.npy"
T8_PACK="$PACK_DIR/video_pack_128_t8.npy"
OUT="out_01_tel_axisx_T8_local"   # matches mutagen ignore out_*/ -> stays local

# 1) slice T16 -> first-8-frames pack (idempotent)
if [ ! -f "$T8_PACK" ]; then
  "$PY" - "$SRC_PACK" "$T8_PACK" <<'PYEOF'
import sys, os, json, numpy as np
src, dst = sys.argv[1], sys.argv[2]
a = np.load(src)                         # (N,16,H,W,3) uint8
a8 = a[:, :8].copy()                     # first 8 frames -> preserve t0/v0
np.save(dst, a8)
m = src + ".meta.json"
if os.path.exists(m):
    meta = json.load(open(m))
    meta["t"] = 8
    meta["_sliced_from"] = os.path.basename(src) + " [:, :8] (first-8 to keep t0/v0)"
    json.dump(meta, open(dst + ".meta.json", "w"), indent=2)
print(f"sliced {a.shape} -> {a8.shape} saved {dst}")
PYEOF
fi

# 2) train. gpu_guard fails open locally (no ws-status); preset CUDA_VISIBLE_DEVICES
#    to short-circuit its 18GB free check. quota knobs disabled for local run.
exec env CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PY" train_video.py \
    --pack "$T8_PACK" --out "$OUT" \
    --res 128 --frames 8 --dim 64 --dim_mults 1 2 4 8 \
    --epochs "${EPOCHS:-200}" --batch 1 --grad_accum 4 \
    --sample_every 50 --ckpt_every 10 \
    --quota_floor_hours 0 --quota_stop_secs 0 --max_my_gpus 8 \
    --resume auto
