"""Pack a dataset_gen run into a single training tensor (the video "pack").

NOT a cache -- it is the whole dataset's videos, resized to RES x RES and stacked
into one (N,T,RES,RES,3) uint8 array, ready to memory-map for diffusion training.
By default it is written INSIDE the dataset run dir (video_pack_<res>.npy), so the
pack lives with the dataset that produced it -- "which dataset was this trained
on?" is answered by location, and that dir already carries config/manifest/README.

Requires homogeneous T across the dataset (single stacked array); asserts it.

  python pack_dataset.py --data_dir ../outputs/dataset_gen/01_tel_axisx_rest_T16 --res 128
      # -> ../outputs/dataset_gen/01_tel_axisx_rest_T16/video_pack_128.npy (+ .meta.json)

Output (default beside the dataset):
    <data_dir>/video_pack_<res>.npy        uint8 (N, T, RES, RES, 3)
    <data_dir>/video_pack_<res>.meta.json  {res,n,t,data_dir,E,v0,T,camera,ids}
"""
import argparse
import glob
import json
import os

import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="a dataset_gen run dir (has sample_*/)")
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--out", default=None,
                    help="pack .npy path (default: <data_dir>/video_pack_<res>.npy)")
    args = ap.parse_args()
    out = args.out or os.path.join(args.data_dir, f"video_pack_{args.res}.npy")

    dirs = sorted(glob.glob(os.path.join(args.data_dir, "sample_*")))
    assert dirs, f"no samples under {args.data_dir}"
    vids, Es, v0s, Ts, cams, ids = [], [], [], [], [], []
    T0 = None
    for d in dirs:
        v = np.load(os.path.join(d, "video.npy"))  # (T,H,W,3) uint8
        T = v.shape[0]
        if T0 is None:
            T0 = T
        assert T == T0, (f"heterogeneous T ({T} vs {T0}) at {d}: this pack format "
                         f"needs uniform T (variable-T needs a per-clip pack + batch=1).")
        frames = np.empty((T, args.res, args.res, 3), dtype=np.uint8)
        for t in range(T):
            frames[t] = np.asarray(
                Image.fromarray(v[t]).resize((args.res, args.res), Image.BILINEAR))
        vids.append(frames)
        s = json.load(open(os.path.join(d, "sample.json")))
        Es.append(s.get("E")); v0s.append(s.get("v0")); Ts.append(s.get("T"))
        cams.append(s.get("camera")); ids.append(os.path.basename(d))

    arr = np.stack(vids, 0)  # (N,T,RES,RES,3)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.save(out, arr)
    with open(out + ".meta.json", "w") as f:
        json.dump({"res": args.res, "n": int(arr.shape[0]), "t": int(arr.shape[1]),
                   "data_dir": os.path.abspath(args.data_dir),
                   "E": Es, "v0": v0s, "T": Ts, "camera": cams, "ids": ids}, f)
    print(f"saved {out}  shape {arr.shape}  dtype {arr.dtype}  size {arr.nbytes/1e6:.1f} MB")


if __name__ == "__main__":
    main()
