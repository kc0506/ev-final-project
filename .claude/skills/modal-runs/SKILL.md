---
name: modal-runs
description: 用 Modal 把 reuse_mpm / gic 的 GPU 工作 offload 到雲端的操作指南。當你要在 Modal 上跑東西、改 infra/modal/ 的 image、加新 entrypoint、複製/擴充某個 conda env、處理 Modal volume、或 debug image build 失敗時適用。
---

把 `reuse_mpm`（physdreamer env）與 `../gic`（gic env）的 GPU 工作 offload 到 Modal。兩個 image 已建好且端到端驗證過（見 [[modal-gpu-offload]] memory）。這份 skill 是「怎麼用 + 怎麼擴 + 怎麼 debug」的固化指南。**改 image 或加 entrypoint 前先讀這裡，別重踩已知的坑。**

## 0. 前置

- Modal client 裝在獨立 mamba env `modal`（py3.12）：`/tmp2/b10401006/.symlinks/miniforge3/envs/modal/bin/modal`。token 已設好（`~/.modal.toml`, profile `kc0506`）。
- **不要**把 modal 裝進 physdreamer/gic（污染脆弱 stack）或 global（mise py3.14 沒 wheel）。
- 所有 Modal 檔案在 `infra/modal/`。**這個目錄不可叫 `modal/`**，會 shadow `import modal`。

## 1. 快速跑（現成的）

```bash
MODAL=/tmp2/b10401006/.symlinks/miniforge3/envs/modal/bin/modal

# import smoke（確認 env 完整、最便宜）
$MODAL run infra/modal/gic.py::versions
$MODAL run infra/modal/physdreamer.py::versions

# 真實 entrypoint
$MODAL run infra/modal/gic.py::fit_telephone              # gic traj fit (A10)
$MODAL run infra/modal/physdreamer.py::forward_telephone  # reuse_mpm forward_gen (A100)

# 換 GPU（啟動時覆寫；見 §5）
GENPHYS_GPU=H100 $MODAL run infra/modal/gic.py::fit_telephone

# 拉結果回本機
$MODAL volume get pd-out fwd_telephone/video.mp4 /tmp/x.mp4 --force
```

第一次跑會 build image（gic ~20min、physdreamer ~15min，多在編 CUDA 擴充）；之後 layer 被 cache，秒級啟動。

## 2. 核心架構：為什麼是「獨立 env + subprocess」

**Modal 的 in-container runtime 需要 Python ≥3.10，但兩個 env 都是 py3.9。** 所以：
- `modal.Image.micromamba()` 的 base env 保持 Modal 相容的 py（3.13）。
- py3.9 的完整 stack 裝進**獨立的** conda env（`/opt/conda/envs/gic` 或 `/opt/conda/envs/pd`）。
- `@app.function` 在 base py 上跑，**subprocess 呼叫** `/opt/conda/envs/<env>/bin/python` 跑真正的 code。

code 用 `.add_local_dir(...)` 在 runtime mount（不 bake），所以改 code 不用重 build。

## 3. 六條硬規則（image build 的坑，全踩過）

1. **build CUDA ext 用 `micromamba run -n <env> ...`**，不是 abs-path pip。沒 activate 的話 git/nvcc 不在 PATH、torch cpp_extension 的 `which g++` 會失敗（conda 編譯器叫 `x86_64-conda-linux-gnu-g++`，activation 才設 `CC`/`CXX`）。一律加 `--no-build-isolation`（build 時要看得到 torch）。
   - 有 glm submodule 的 rasterizer（graphdeco 原版）要 `git clone --recursive`（pip git+ 不抓 submodule），整串包進 `micromamba run -n <env> bash -c "git clone ... && pip install --no-build-isolation ."`。
2. **subprocess 要清掉 `PYTHONPATH=/pkg`**：Modal runtime 把自己的 `/pkg` 塞進 PYTHONPATH，被 3.9 subprocess 繼承後啟動就 import Modal、撞 py3.10 guard。subprocess 的 `env` 設 `PYTHONPATH=/root/<repo>`。
3. **`.add_local_*` 必須最後**，後面不能再有 build step（含 `.env()`、`.run_commands()`、`.apt_install()`）。要保 cache 就把新 dep 層加在重層（conda/ext）**之後**、`.add_local_*` 之前。
4. **arch list 跟 CUDA 對齊**：gic（cu121）`TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0"`；physdreamer（cu118）`"7.0 7.5 8.0 8.6"`（**不含** 8.9/9.0，cu118/torch2.0 不支援 L4/Hopper）。
5. **隱性 system / lazy deps**（environment.yml 跟 freeze 都看不到）：`ffmpeg`（mediapy 寫 mp4，apt）、`libgl1`+`libglib2.0-0`（open3d/cv2，apt）、`kmeans-gpu==0.0.5`（`local_utils` build cache 時 lazy import）。掃 lazy import 要看 function 內層的 `import`，不只 top-level。
6. **scene cache 寫不進唯讀 mount**：`/root/<repo>` 是唯讀。entrypoint 若會寫 cache/輸出，導向 volume：gic 用 `--out /out/...`、reuse_mpm forward_gen 用 `--scene.cache-path /out/...`。

## 4. 複製 / 擴充一個 conda env（env-truth 方法論）

要把某個本機 conda env 搬上 Modal，**先判斷 env 乾不乾淨**：

- **乾淨 env（如 gic）→ `pip freeze` 是真相**。`pip freeze` 後濾掉 `@ file://`（conda 提供、transitive 會帶）、torch 三件（另裝）、git ext（另 build），剩下當 requirements。
- **髒的共用 env（如 physdreamer）→ 用 repo 自己維護的 `environment.yml`**。freeze 會混進無關套件且內部衝突（physdreamer 的 freeze：jaxtyping 硬 pin typeguard==2.13.3 撞 tyro 的 4.x → ResolutionImpossible）。
- **environment.yml 也不夠**：package code 常 import 沒宣告的 runtime dep。`grep` 真正會被 import 的模組（含 function 內 lazy import）找出來，**對齊本機 env 的確切版本**（`pip show` / `importlib.metadata.version`）。physdreamer 補的：opencv-python==4.8.1.78 / decord==0.6.0 / omegaconf==2.1.1 / scikit-learn==1.3.2 / trimesh==4.12.2 / pymeshlab==2023.12.post1 / jaxtyping==0.2.28（`--no-deps`）/ kmeans-gpu==0.0.5。**全部釘 numpy<2**（最新 opencv 拉 numpy>=2 會破 torch2.0/warp）。
- 衝突但 runtime 能共存的（如 jaxtyping vs typeguard）用 `pip install --no-deps`，複製本機「逐步裝成」的狀態。

非 PyPI 的本機 package（`physdreamer` + `local_utils`）：純 sys.path，不用 pip install。把 checkout `.add_local_dir` 進 image、設環境變數 `PHYSDREAMER_ROOT` 指過去，`reuse_mpm/_env.py` 自己接。

## 5. GPU 與 peak mem

- GPU 用 `GENPHYS_GPU` 環境變數選；smoke 一律 T4（最便宜，只跑 import）。
- **physdreamer（cu118）上限 A100**，別碰 H100/H200/B200（Hopper/Blackwell + 舊 cu118 不相容）。**gic（cu121）到 H100 OK**，B200 不行（sm_100 要 cu12.8）。
- 行情（$/hr）：T4 0.59 / **L4 0.80** / **A10 1.10** / L40S 1.95 / A100-40 2.10 / A100-80 2.50 / H100 3.95。甜蜜點 A10/L4（24GB）。
- **peak mem 要量整卡**（`nvidia-smi --query-gpu=memory.used`），不是 `torch.cuda.max_memory_allocated`——gic 的 taichi `ti_mem_frac=0.3` 直接預留 30% 顯卡、torch metric 量不到。`forward_telephone` / `fit_telephone` 都內建 nvidia-smi 取樣 thread，回報整卡峰值。
- 實測：gic telephone traj 7.4GB（多半 taichi 預留）、physdreamer forward_gen telephone 5.8GB（cache 載入）/8.9GB（cache build）。都 <10GB → A10/L4 夠。

## 6. Volumes

| volume | 掛載點 | 內容 |
|---|---|---|
| `gic-data` | `/data` | scene caches（/telephone... + /scene_cache/*）|
| `gic-out` | `/out` | gic run 輸出 |
| `pd-data` | `/tmp2/b10401006/PhysDreamer/data/physics_dreamer` | telephone dataset（同**寫死的絕對路徑**掛，解 cache 的 hardcoded `dataset_dir`）|
| `pd-out` | `/out` | forward_gen 輸出 + scene cache |

```bash
$MODAL volume create <name>
$MODAL volume put <name> <local> <remote>          # 上傳；目錄也可
$MODAL volume ls <name> [path]
$MODAL volume get <name> <remote> <local> --force  # 下載
```

**dataset_dir caveat**：scene cache `.pt` 裡 `dataset_dir` 是寫死絕對路徑。最省事解法是把資料掛在**同一絕對路徑**（如上 pd-data），零改 code、零 patch cache。

## 7. 加一個新 entrypoint

複製 `gic.py` / `physdreamer.py` 裡現成 function 的形狀：

```python
@app.function(image=<env>_image, gpu=GPU_RUN,
              volumes={"/data": data_vol, "/out": out_vol}, timeout=...)
def my_run() -> str:
    import subprocess
    cmd = [ENV_PY, "-m", "reuse_mpm.<entrypoint>", "--...", "--out", "/out/<run>"]
    r = subprocess.run(cmd, cwd="/root/<repo>", capture_output=True, text=True,
                       env=_pd_env())   # 清 PYTHONPATH！見規則 2
    out_vol.commit()                    # 寫完要 commit volume 才持久化
    ...
```

要點：subprocess 跑 `<env>/bin/python`、`cwd` 設 mount 的 repo 根、`env` 清 PYTHONPATH、輸出導向 `/out`、結束 `out_vol.commit()`。要實測 mem 就照現成的 nvidia-smi 取樣 thread 包一層。

## 8. Debug build 的節奏

image 是 layered + cached：把**穩定/重的層放底**（conda env、torch、CUDA ext），**易變/易錯的放上面**（小 dep、apt、code mount）。一個 layer 失敗，下面全 cache，改完重跑只重編那層之後。所以遇錯**逐層修、逐次重跑**很便宜——一次別想修全部。

跑法：`$MODAL run ... &`（背景）+ 盯 log（grep `OK|FAIL|Traceback|error:|Built image`）。CUDA ext（尤其 pytorch3d）編譯 10-20min 是正常的，別誤判 hang。
