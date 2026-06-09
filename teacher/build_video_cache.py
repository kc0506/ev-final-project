"""
Precompute a downsampled video cache for the phase-2 video-diffusion overfit test.

Reads each sample's video.npy (8, 576, 1024, 3) uint8, resizes every frame to
RES x RES, stacks into one array (N, T, RES, RES, 3) uint8, and saves it plus
the per-sample E (for later identifiability checks, NOT used by the
unconditional video model).

Output:
    cache/video_{RES}.npy   uint8 (N, T, RES, RES, 3)
    cache/video_meta.json   {res, n, t, E: [...], ids: [...]}
"""
import glob
import json
import os

import numpy as np
from PIL import Image

DATA_DIR = "/tmp2/b10401006/ev-project/generative-phys/outputs/dataset_telephone_256"
OUT_DIR = "/tmp2/b10401006/ev-project/generative-phys/teacher/cache"
RES = 128


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    dirs = sorted(glob.glob(os.path.join(DATA_DIR, "sample_*")))
    assert dirs, f"no samples under {DATA_DIR}"
    vids, Es, ids = [], [], []
    for d in dirs:
        v = np.load(os.path.join(d, "video.npy"))  # (T,H,W,3) uint8
        T = v.shape[0]
        frames = np.empty((T, RES, RES, 3), dtype=np.uint8)
        for t in range(T):
            frames[t] = np.asarray(
                Image.fromarray(v[t]).resize((RES, RES), Image.BILINEAR)
            )
        vids.append(frames)
        with open(os.path.join(d, "sample.json")) as f:
            Es.append(json.load(f)["E"])
        ids.append(os.path.basename(d))
    arr = np.stack(vids, 0)  # (N,T,RES,RES,3)
    out_npy = os.path.join(OUT_DIR, f"video_{RES}.npy")
    np.save(out_npy, arr)
    with open(os.path.join(OUT_DIR, "video_meta.json"), "w") as f:
        json.dump({"res": RES, "n": arr.shape[0], "t": int(arr.shape[1]),
                   "E": Es, "ids": ids}, f)
    print(f"saved {out_npy}  shape {arr.shape}  dtype {arr.dtype}  "
          f"size {arr.nbytes/1e6:.1f} MB")


if __name__ == "__main__":
    main()
