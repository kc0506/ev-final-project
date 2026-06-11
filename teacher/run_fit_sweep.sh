#!/usr/bin/env bash
# Sweep teacher video-diffusion (dim=64) over frames x res on the LOCAL GPU,
# one config per process (clean CUDA mem each), printing peak GPU mem per config.
set -u
cd "$(dirname "$0")"
PY=.venv/bin/python
echo "device: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
printf '%-8s %-6s %-10s %-9s %s\n' frames res params_M peak_gb status
for res in 64 96 128; do
  for frames in 8 12 16; do
    line=$("$PY" probe_fit.py --frames "$frames" --res "$res" --dim 64 \
             --dim_mults 1 2 4 8 --batch 1 --steps 3 2>/dev/null)
    # parse the JSON line with the same interpreter
    echo "$line" | "$PY" -c '
import sys,json
d=json.load(sys.stdin)
c=d["config"]
print("%-8s %-6s %-10s %-9s %s" % (c["frames"],c["res"],d.get("params_M"),d.get("peak_gb"),d.get("status")+(" "+d.get("error","") if d["status"]!="ok" else "")))
' 2>/dev/null || echo "  (no output for frames=$frames res=$res — $line)"
  done
done
