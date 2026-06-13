# F0 sys-id：loss-design 與 F0-training 一輪總結

_2026-06-12/13。全部在 **warp(physdreamer env)前向 + 有限差分** 的 block 實驗室,warp-self
roundtrip(GT 與 fit 同模擬器,無 model-mismatch floor)。輸出根目錄 `outputs/explore/`。_

---

## TL;DR

1. **E 進入運動有兩條通道,由激發是「位移控制」還是「力控制」決定**:位移→只剩**頻率**通道(time-domain beat 飽和,需 spectral);力→**振幅**通道(time-domain 直接讀得到)。沒有萬用 loss,該用哪個由通道決定。
2. **spectral loss 解了 beat plateau**(從下方收、at-GT 穩),但有**兩個獨立病灶**:低 E 側顛簸 local-bump(位置隨 K 移),高 E 側 **Nyquist 混疊假盆地**(logE≈5.47 牆、5.95 鬼影)。
3. **combined(time+λ·spectral)沒有 scene-agnostic 的固定 λ**(工作窗空);但**「各跑各的、再用跨通道一致性挑贏家」可行**。
4. **F0 training 機制驗證通過**(FD):global homogeneous V0=expm(S) 6-DOF 乾淨 recover;coarse field(I+∇u 解析基底 15-DOF)recover 目標模態 + map 出一條淺簡併。
5. **方法論大 caveat:F0 這條全程有限差分,MPM autograd 一次都沒用**(scalar/低 DOF 可行)。field(MLP)+真資料 → 必須 autograd → gic。

---

## 1. 中心框架:振幅通道 vs 頻率通道(位移 vs 力控制)

| 激發 | 控制型態 | E 走的通道 | 機制 | 該用 loss |
|------|---------|-----------|------|----------|
| pure F0-release(g=0,無接觸) | **位移控制** | **頻率**(∝√E) | 振幅被 F0 釘死、與 E 無關(SHO 振幅=初始位移);time-domain 兩異頻同振幅軌跡距離上界 2A → **beat 飽和成平台** | **spectral** |
| drop(重力+碰撞) | **力控制** | **振幅** | 形變量 = 力/勁度 ∝ 1/E;接觸反力讓振幅帶 E | **time-domain**(3D MSE,本就 shape-agnostic) |

- **drop 實證**:`exp2_drop_time_spectral_init35` — time_L2 從 init 3.5 **復活收到 4.53**(對照 pure-release 卡 3.6);spectral 反而敗(width 訊號被撞擊弄濁)。= **PAC-NeRF 對帳**:他們靠重力/碰撞場景,所以 time/image loss 好學;我們用 release 把振幅通道關掉了,才需要 spectral。
- **互補定理**:release 上 time 平/spectral 收;drop 上 time 收/spectral 平。
- 來源:`outputs/explore/f0_block_squeeze_sweep/`(R=2/3/5 robust)、`outputs/explore/f0_fit_case/exp2_*`、`f0_loss_landscape/rel_drop/`。

## 2. spectral loss:運作、進度、caveats

**怎麼運作**:取一個觀測量(這裡用 `width(t)`=x-extent)→ FFT → 比較頻譜。**注意不能用 naïve `‖FFT_pred−FFT_gt‖²`(峰不重疊時也飽和成常數)**;有效的是「頻率值」——週期 ∝1/√E **單調不飽和**,所以 spectral-L2(在重疊區)/centroid 有梯度。實測 spectral-L2 從下方 init 乾淨收(`f0_fit_case` exp1:init 4.75→4.503)。

**已驗證進度**:在 R=2/3/5(換 F0 振幅)都複製「spectral 收、time 敗」;at-GT 穩定性是最乾淨判別(spectral 4.5→4.48 穩;time 4.5→4.27 漂走 = GT 非其不動點)。

**完整地形圖**(`f0_spectral_K_probe/`):
- **軟側(< GT)**:顛簸 + 一個 ~1.5% 的 **local bump**,位置隨 K 移動(K=18 bump@3.6 在 init 3.5 上方→把 init 往下反射→**stall**;K=32 bump@3.44 在下方→收)。**「argmin 在 GT」是廢指標**(loss-vs-GT 在 GT=0,argmin 恆在 GT);要看「**從 init 往 GT 的斜率/capture radius**」。
- **高 E 側 Nyquist**:週期降到 **logE≈5.47(=2 frame)觸 Nyquist 牆**,之上 measured period **折返混疊**(6.0→週期 9.0),spectral loss 非單調、長出 **logE≈5.95 的混疊假盆地**(= GT 頻率的鬼影)。**init 5.5→5.94 就是掉進它。** Nyquist 與 K 無關(取樣率效應),修法=提高 frame rate(降 delta_t)推高牆。

**caveats**:(a) 觀測量 `width` 只對對稱呼吸場景成立,**不對稱場景(squeeze)width 失效 → spectral 推廣未解**;(b) 脆弱——收不收取決於 init 落在最近 bump 哪一側;(c) 需配 sampling-init / coarse-to-fine 跨過 bump 並只在 [軟側~5.47] 區間取樣。

## 3. combined loss:固定 λ 無萬用解,但「挑贏家」可行

`L = time/c_t + λ·spectral/c_s`(固定 ref 歸一)。**工作窗是空的**(`f0_fit_case/exp{6,7}_*`、`exp_band_*`):

| λ | release(3.5→) | drop(3.5→) |
|---|---|---|
| 1.0 | 4.49 ✓ | 3.87 ✗(spectral 在 drop 的 4.0–4.1 ridge 擠出 3.9 局部井困住 time;陷阱盆地涵蓋 init≤4.0) |
| 0.3 | 3.76 ✗ | 3.89 ✗ |
| 0.1 | 3.59 ✗(spectral 太弱拉不動) | 4.56 ✓ |

release 要高 λ、drop 要低 λ → 相反,沒有單一 λ 通吃。**「優化」combined 失敗(繼承 ridge+beat 兩病);但「選擇」combined 成功**:各跑單 loss fit,挑 `L_time+L_spec` 最小的收斂候選(真 E 唯一讓兩者≈0,卡錯通道的 E 必有一個高)。快取驗證:release 挑中 4.50、drop 挑中 4.53,兩場景都 recover GT、無需知道場景或調 λ。

## 4. 激發強度 gate capture radius(squeeze,與通道正交)

`f0_fit_case/exp8_squeeze_K24`:squeeze(不對稱下壓 vs floor,**力通道**)maxdev 僅 0.073(弱)→ **time 和 spectral 都只在 GT 穩、兩側無 capture radius**。但 landscape(`f0_loss_landscape/squeeze/`)顯示 **time 的斜率隨 K 變**:K=12 Δ+0.61、K=24 Δ+0.06 → **不是激發太弱,是 K=24 太長把早期振幅訊號用後段 beat 稀釋掉**。教訓:**capture radius 由激發強度 + 正確 K 共同 gate**,與「通道」正交;弱激發/錯 K 即使對的通道也收不動。

## 5. 重力惰性(free-fall confirm)✅

`f0_fit_case/exp{3,4,5}_*`:**自由落體(重力、無接觸)對 sys-id 零貢獻**——不形變、不帶 E。鐵證:freefall 的 `width(t)` 與 g=0 **位元相同**(Δ 0.03%),且 **release-K18 ≡ freefall-K18 的 fit 位元相同**。所以 drop 讓 time 復活的活性成分是**接觸反力,不是重力**;重力只負責把物體送去撞地板。freefall-K18 spectral 之所以也敗純粹是 K 太短(exp4 K=32 下 spectral 在重力場照樣收到 4.50)。

## 6. K 對 landscape:量變非質變

`f0_spectral_K_probe`(K=8…48)+ `f0_loss_landscape`(K=12/18/24/32):**各 K 的 landscape 長相相似**(全域 min 都在 GT、軟側顛簸、高側 Nyquist),**K 只造成微擾**——主要影響是:(a) 軟側 local-bump 的**位置**隨 K 平移(決定特定 init 收不收);(b) Nyquist 是 K-independent。**沒有「某個 K 才質性改變 landscape」**。實務:長 K 只把井底磨尖(near-GT 精度↑)、把 bump 推離典型 init,但 capture-radius 的平台問題不因 K 而消失。

## 7. 方法論 caveat:F0 線全程有限差分,無 MPM autograd ⚠️

**這整輪沒用過 MPM autograd。** 梯度全是 `(L(θ+ε)−L(θ−ε))/2ε`,warp 前向跑多次(`no_grad`/`requires_grad=False`)。warp 的可微反傳路徑從未呼叫。
- **為什麼**:scalar E(1 DOF)、global-S(6)、coarse(15)都低維,FD(2×DOF evals/iter)精確夠用、穩健、繞過長時域 backward 不穩。對「哪個 loss landscape 可用」這問題前向+FD 就夠。
- **含意**:(1) 這些結果**沒驗證過梯度路徑**;對低 DOF FD≈autograd。(2) **真訓練要 autograd**(field 高維 FD 不可行);而 **spectral 需長 K(32)→ autograd 反傳穿 ~2048 步 → 正好踩長時域 backward 不穩**,這風險**未測**。(3) **field(MLP)/真 recovery → 必須搬 gic**(有 `_read_F0_grad` bridge);warp-self 的 ~−0.02 dex 誤差偏樂觀(無 model floor)。

## 8. 手動構造的 scene + release 條件(catalog)

統一在 `reuse_mpm/explore/_block.py` 的 `Scene(name)`,`SCENES[name]=(f0方法, release重力?, release地板?)`:

| scene | F0 怎麼來 | release | 用途 |
|-------|----------|---------|------|
| `release` | 兩端 x-pull `pull_frames`(速 0.5、grip_half_x 0.045)→ snapshot 非均勻 F0 | g=0,無地板 | 位移控制/頻率通道主場 |
| `drop` | 同 pull | g=−9.8,**地板 z=0.25(gap 0.05)** slip | 力控制/振幅通道;碰撞(min_z 略穿 floor 0.024 可接受) |
| `freefall` | 同 pull | g=−9.8,**無地板** | 證明重力惰性 |
| `squeeze` | 右側(x~0.6)往下壓 vs 地板 `push_frames`(速 0.45)→ 不對稱 F0 | g=0,地板 z=z_base | 不對稱/力通道;弱激發 |
| `uniform` | **直接注入** `F0=expm(S)` + compatible 仿射位移(繞中心) | g=0,無地板 | F0-training self-consistent GT |
| (grad-u viz) | **已知** `u_y=A·sin(πξx)` 半正弦彎曲 → `F0=I+∇u`(`F0[1,0]` 隨 x 變) | g=0 | 非均勻 F0 的已知-u GT(`f0_gradu_viz`) |

關鍵幾何:`half=(0.18,0.08,0.14)`、`z_base=0.30`、grid 32、substep 64、delta_t 1/30、jelly、nu 0.3、density 2000。warp wall-clamp 在 2dx≈0.063(drop/freefall 落距要留餘裕)。

## 9. F0 parametrization 選項 + 訓練結果

- **expm(S)(log-Euclidean,左拉伸 V0)**:S 對稱 6-DOF、恆 SPD、S=0=rest;旋轉是 dynamics gauge(估不到)。**用於 global/smoke**。
  - 結果 `f0_train_S/globalS_run`:GT 均勻 `S_gt=(0.2,−0.1,−0.1,0.05,0,0)`、E 固定 GT、time_L2、FD-6 → **6 分量全 recover、|err|<0.011、loss 2.76e-4→7.6e-7**;連 GT=0 的 xz/yz 都穩在 0 → 此激發下 6-DOF 全可辨識。(E 固定時 S 改變振幅/形狀非頻率 → time_L2 就好,無 beat。)
- **I + ∇u(compatible field)**:參數化位移場 u(x),F0=I+∇u **保證 compatible**(真實形變)、降 DOF(3/particle)、天然平滑。
  - **coarse(解析基底,FD,warp)** `f0_train_ufield/coarse_bend`:基底 `[ξx,ξy,ξz,B(ξx),B(ξx)·ξy]×3 分量=15`,GT 只有 `uy:bend=0.05`。結果 **bend recover 到 0.048、13/14 distractor 回 0、loss→1.38e-5**;唯一 `uy:bend*y` 卡在 −0.017 = **與目標 bend 淺簡併**(傾斜 bend 與純 bend 動力學近同)→ 這是 coarse-field 可辨識子空間的第一張地圖。
  - **fine(MLP,autograd,gic)** 未做:plain 小 MLP(spectral bias=免費平滑正則,符合平滑 F0 prior)、xyz 先 normalize、**Fourier encoding 預設不要**(只在局部尖銳細節 underfit 時加幾個低頻)、末層初始化 ~0、F0=I+∇u 用 torch jacobian。dL/dθ 穿 MPM → gic autograd。field 欠定 → 靠平滑 prior 撐。

DOF 階梯:**global(6)✅ → coarse field(15)✅ → fine field(MLP,gic)🔜**。

## 10. 工具 / entrypoints

- `_block.py`:`Scene` 共用機制(geometry/build/setE/F0-snapshot/rollout)。**加 scene = SCENES 加一筆 + 其 f0 方法**,fit/landscape 都自動吃。`rollout_F0(x0,F0,logE,K)` 吃任意 F0(供 F0-fit)。
- `f0_fit_case.py`:單-case scalar-E fit(scene/loss/init 參數化);overlay 固定班底;combined loss。
- `f0_loss_landscape.py`:time/spectral/combined landscape over {scene}×{K}×{E};**快取 loss 成分 → reweight 離線即時**(改 λ 不用重跑 GPU)。
- `f0_spectral_K_probe.py`:spectral 的 K/Nyquist 地形(週期折返、混疊盆地)。
- `f0_train_S.py`:global expm(S) 6-DOF FD fit。
- `f0_train_ufield.py`:coarse u-field 解析基底 FD fit。
- `f0_gradu_viz.py`:已知-u 非均勻 F0 的 forward viz。

## 11. GPU / quota 教訓(過程中)

- 雙卡扣率:`閒置<2 + 同使用者雙卡 >10min → (N+0.2)×N`(4.4×+)。**使用者已在用某卡時,我的小 job 要併到同一張卡**(pin `CUDA_VISIBLE_DEVICES`),別讓 `pick_free_gpu` 散成雙卡。長 run 加 `timeout` 硬上限。
- **loss/logE 曲線會騙**:log 軸 autoscale 把 3% FD 噪音放大成「大跌」;且「參數收到哪 ≠ 運動對」→ 每個 fit 自動產 GT-vs-收斂 的 3d+triplane overlay 當固定班底。

---

## Open / next

- **fine field = MLP + MPM autograd(gic)**:質的跳躍(高 DOF、cross-sim model floor、長 K backward 穩定性風險)。
- spectral 的 **shape-agnostic 觀測量**(width 只對對稱;PCA-extent / 隱式觀測 / Resonance4D 的 log-mag+phase)——這才是 spectral 真正能不能推廣的關鍵。
- E↔F0 **joint**(淺 E·strain 簡併 ridge)、cross-sim、真資料。
- coarse-field 的 `uy:bend*y` 簡併:加正則 / 換基底是否消得掉。
