---
name: pipeline-conventions
description: reuse_mpm pipeline 的 repo convention 與 refactor 紀律。當你要新增/修改 entrypoint、碰 RunDir/輸出目錄、改 config.py、寫 explore 診斷腳本、或處理 scene cache / 決定論 / provenance 時適用。
---

這份 skill 把 reuse_mpm 這個 research pipeline（distill video-diffusion teacher → MPM 物理參數 generative model）的 code convention 與兩輪 refactor 的教訓固化下來。新增或改動 pipeline code 前先讀這裡，不要重新發明已經定好的契約。

## 核心心法：research code != sloppy code

品質重點**不在設計模式**，而在這四條（用戶原話）：
1. **明確的 input/output** — 可重現的結構化結果，一個輸出目錄 == 這次 run 的全部。
2. **不亂噴 defaults** — 不要把參數/預設值寫死在某個 function 最底層。需要的 default 往上拉到 config dataclass。
3. **區分一次性探索 vs canonical** — `explore/` 是拋棄式診斷，正式 entrypoint 是會被反覆使用的。兩者慣例不同（見下）。
4. **以後會用到的參數往上拉** — 預期會調的東西放進 config，不要埋在實作裡。

附帶硬性要求（來自 `research-discipline`）：每個 function 有 return type hint；tensor 一律註解形狀；docstring 註明 input/output shapes。`CLAUDE.md`：import 無特殊理由一律 module top-level。

## 輸出契約：RunDir + 輸出樹（`reuse_mpm/run_io.py`）

**一個輸出目錄 == 這次 run 的一切**：config、provenance、輸入快照、產物、事件時間軸、console log。這是整條 pipeline 最重要的契約。

- **輸出樹**：`outputs/<task>/<NN>[_<label>]/`。`<task>` 由 entrypoint 的 `__name__` 經 `task_subpath_from_module` 推導（rename/move 不會漂移；`python -m` 下從 `__main__.__spec__.name` 還原真實 dotted path）。`<NN>` 在該 task dir 內 auto-increment，`ls` 即見 run 順序。**名字不放 timestamp**，時間寫進 `started_at.txt`。
- **RunDir 是唯一的寫入 choke-point**。每個寫方法都會往 `.events.txt` append 一條 timestamped 語意事件（不需要 filesystem watcher）。`finish()` 封存：把任何**繞過** RunDir 方法寫進去的 top-level 檔（`np.save`/`plt.savefig` 經 `.path()`）按 mtime 補登，子目錄折成一行摘要，最後全時間軸排序。
- **config 自動存**：`RunDir.create(__name__, label, out, config=cfg)` 直接把 tyro 解析出的 dataclass 序列化成 `config.json`（schema == dataclass，永不手刻）。run 特有的衍生事實用 `merge_config(**extra)`、`result(...)`、`metrics(...)` 等之後補。
- **declarative 子類**：`ForwardRun`/`RecoverRun`/`DatasetRun` 在 docstring + 具名方法裡**宣告**該 task 產出的 artifact 集合，輸出 schema 集中一處。但**寫入仍是增量的**（不要 buffer 一堆 tensor 到最後才 flush）——declarative ≠ buffer-then-serialize。
- **`copy_in(src, name)`**：把外部檔**複製**進 run dir 做不可變快照（不是 symlink）。用來凍結這次 run 實際用的 scene cache（見決定論一節）。symlink 只用在輸入 ply 這種來源穩定的東西（`link_source_ply`）。
- **`capture_output()`**：`with rd.capture_output():` 包住 run body，用 `os.dup2` 在 **fd 1/2 層級** tee stdout+stderr 進 `console.log`。FD-level 是關鍵——連 taichi/warp/C-extension/subprocess 的輸出都吃得到，不只 Python 的 `print`；pump thread 同時寫檔與終端機（真 tee，終端機照樣即時看到），未捕捉的 traceback 也進 log。

## Config 單一真相源（`reuse_mpm/config.py`）

- **一切可調參數的唯一來源**。CLI 用 **tyro**（dataclass-driven），不要 argparse。entrypoint 形態固定為 `def run(cfg): ...` + `if __name__ == "__main__": run(tyro.cli(XxxConfig))`。
- **`SimConfig`**：num_frames/substep/fps/grid_size/grid_lim/density/material/nu… `grid_size` 由 `SimConfig` 提供（不放 `SceneSpec`），這樣 cache key 與 rollout 永不打架。
- **`SceneSpec` 兩條 disjoint 路徑**（用戶明確偏好 disjoint 而非 flag soup）：要嘛 `preset`（enum，從 `PRESETS` 取 kind+path），要嘛顯式 `path`+`kind`，兩者擇一，`__post_init__` 強制 XOR。consumer 用 `display_name` property 取名。
- **scene 載入唯一入口**：`scene_io.load_from_spec(spec, sim)` 統一 dispatch pd/pg、用 `sim.grid_size`、若 `spec.cache_path` 是 None 就填入解析出的預設路徑（序列化後就記下到底用了哪份 cache）。不要在 entrypoint 裡複製 pd/pg 分支。

## Entrypoint 解剖（forward_gen / train_global_E / dataset_gen）

標準骨架：
```python
def run(cfg: XxxConfig):
    pick_free_gpu()                      # GPU-contended 的 entrypoint 自己挑卡
    from .run_io import XxxRun           # 重依賴 lazy import（見下）
    rd = XxxRun.create(__name__, label, cfg.out, config=cfg)  # auto-save config.json
    with rd.capture_output():            # tee stdout/stderr -> console.log
        scene = load_from_spec(cfg.scene, cfg.sim)
        rd.copy_in(cfg.scene.cache_path, "scene_cache.pt")    # 凍結這次的離散化
        ...
        rd.finish()
    return rd
```
- **lazy import 是 GPU 順序的特例**：torch/CUDA、scene/sim_render 等在 `run()` 內、`pick_free_gpu()` **之後** import，確保挑卡先於 CUDA context 建立。這是 `CLAUDE.md` 「top-level import」的「有特殊理由」例外，刻意為之。
- **body 很長時**（如 dataset_gen 的 sample loop）拆成 `_run(cfg, rd, t0)` helper 再用 `with` 包，避免整塊重排縮排造成巨大 diff。
- `forward_gen` 不自動挑卡（沿用原行為，用 `cfg.scene.device`）；GPU 競爭的 entrypoint 才呼叫 `pick_free_gpu()`。

## explore/ vs canonical（`reuse_mpm/explore/`）

- **一次性診斷腳本**。CLI 仍用 tyro，但 config dataclass **LOCAL** 於該腳本（如 `V0SweepConfig`），**絕不碰 `config.py`**——只**讀** `SceneSpec`/`SimConfig` 來組合。
- 一樣走 RunDir 輸出樹：自動落在 `outputs/explore/<script>/<NN>/`，**不要 `--out`** 手指定，遵循 `run_label`/auto-inc 慣例。
- 設計上要 cheap（no_grad forward、curated 小參數集），目的是「在投 GPU 跑完整 dataset 前先看一眼效果」。

## 決定論 / provenance 教訓（踩過的雷）

- **GPU k-means 下採樣是非決定性的**，`torch.manual_seed(0)` **修不了**（seeded 兩次仍 frozen 628 vs 641）。決定論機制是 **cache（build-once-reuse）**，不是 seed。
- 因此：**每次 run 都 `copy_in` 把 scene cache 固化到 run dir**。共享 cache 會被重建，但「這份 run 用的 cache」永遠留存，事後可完全重現。cache key = `<scene>_ds<downsample>_g<grid>_k<topk>`，從 config 可重新推導同一路徑。
- **robust/幾何 default > data-dependent default**：當離散化會抖動時，用幾何法（`geometric_bottom_slab`：沿某軸的 quantile 取底部 slab 當 anchor）比 data-dependent（moving-based freeze mask）穩。新 BC/mask 預設往這個方向設計。

## Debugging 紀律教訓

- **「我有沒有改壞」不要靠 `git status`/`git diff`**——不能排除壞改動已經被 commit，diff 只會誤導。用 `git log -L`/`git log -S`/`git show` 對著具體函數/符號考古，最近的 commit 全都該看。
- **資訊正確 ≠ 結論正確**。用戶的原話糾正：「你的資訊正確, 但你推倒結論錯誤」。觀察到某現象（如 KNN variation）不等於它就是 root cause；先確認因果再下結論。
- **不確定/非慣用的名詞先確認意思**。例：「爆炸」= 數值穩定性差，不是視覺爆炸；「coarse-to-fine」要問清楚哪個模態/尺度/方法。
- 跑 GPU 路徑用 `physdreamer` conda env（`/tmp2/b10401006/.symlinks/miniforge3/envs/physdreamer/bin/python`），遵守共享 GPU 配額。
- **配額硬閘**：`gpu.pick_free_gpu()` 開頭會 `assert_gpu_quota(min_quota_hours=12)`——剩餘 GPU 額度（`ws-status` parse）低於 12h 直接 `SystemExit`，防止長 run 跑到一半被每日配額機制 kill（白做 + 輸出半殘）。fail-open：讀不到配額只警告不擋。`min_quota_hours=0` 可關閉。不經 `pick_free_gpu` 的 GPU entrypoint（如 `forward_gen`）若要同樣防護，可自行呼叫 `assert_gpu_quota()`。

## 一句話總結

新增 pipeline code 時：tyro config 進 `config.py`（或 explore 的 local config）→ `run(cfg)` 骨架 → `RunDir.create(__name__, …, config=cfg)` → `with rd.capture_output():` → `load_from_spec` + `copy_in` cache → 增量寫 artifact → `finish()`。讓「一個輸出目錄 == 一次 run 的全部」這條契約成立，其餘自然到位。
