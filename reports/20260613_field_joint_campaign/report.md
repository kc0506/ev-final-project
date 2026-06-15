# Field & Joint sys-id on gic — campaign report (2026-06-12〜13)

接續 `20260611_gic_anchor`（Q1 cross-sim）與 `20260612_gic_q2`（Q2 global E + v0）。
那兩份已證明 gic（NeurIPS'24 GIC，taichi MPM、全段 BPTT）在我們的 telephone scenario
（heavy-anchor freeze BC、gravity off、置中 norm v2 cache）下，scalar E 與 xy-平面 v0 可恢復。
**本報告涵蓋之後的延伸**：優化教訓 → 影像監督（視角極限與 multiview 解）→ v0 field →
joint v0+E → E field → 各種 non-uniform GT（ramp / mid / bend / circular）→ 跨場景移植。

所有程式在 `../gic/`（untracked）：核心 `roundtrip_ours_scene.py`（AnchoredEstimator + 場注入 +
train_ours + joint/alt 流程 + 視覺化）、`v0_field_ours.py`（V0VoxelField / EVoxelField）、
`efield_fit.py`（E-field + 雙場 + circular）、`multiobs_fit.py`（多觀測共享 E）、
`image_fit_ours.py`（影像監督 + multiview）、`make_panel.py`、`loss_landscape.py`。
數字來源：各 run 的 `output/{ours_telephone,ours_efield,ours_carnations}/<tag>/result.json`。

---

## 1. 優化教訓（本輪真正的可轉移產物）

global-E 本身 grid search 即可解；價值在梯度鏈的優化教訓，後續 field 學習全靠它們。

1. **loss 收斂 ≠ 參數收斂**。多個 case（v0only_yz +48%、alt 卡死、joint 早停截殺）都是 loss
   已平台但參數沒到。判收斂只認梯度不動點 + 視覺軌跡，不讀 loss 曲線數字。
2. **兩段式 lr**：粗段 lr 0.2 穿越 decade、第一次 plateau 回跳 best + 平坦 fine lr 0.02 精修。
   單一指數 schedule 兩頭不討好；均勻 lr/4 餓死（+21.5%）。
3. **early-stop patience 必須 ≥ 震盪週期**（~16–30 iter）；momentum + 大 lr = 欠阻尼震盪。
4. **參數移動量早停**（param_stop_tol）：純 loss-plateau 會殺掉慢爬的分量。joint_initE4 的
   E 在回升途中被 loss-plateau 截殺即此（後改成監看全參數、E 用 log 空間）。
5. **best-of-data-loss 在有 regularizer 時失效**：TV 在 data-loss 看不見的子空間持續改善場，
   `final` 比 `best` 好 3×（v0-field res4 TV：5.2%→1.7%）。匯出慣例該用 final 或 best-of-total。
6. **輸出設計**：易誤導的指標（全軸 relL2、p95、mean-vec err）一律降級到 json；headline 給
   per-axis err、xy-only（可觀測子空間）。數字會錨定第一判斷，靠紀律抵抗不如預設就乾淨。

---

## 2. vz / 耦合 / landscape（為何 z 難學）

- **strip landscape**：`Evy`（v0_y × logE，4f 窗）是窄長條形（v0 梯度尺度 ≫ E），不是橢圓。
  含義：固定 E 下 v0 幾乎都能滑到谷底；E 錯一個 decade，最優 v0_y 只移 10–20%。
- **vz 死區**：z 是電話線長軸，沿長軸滑動對 chamfer 近不可觀測；v0_z 梯度尺度比 v0_y 小一量級。
  初判「chamfer null space」被 landscape 推翻（argmin 在 GT），真因是梯度尺度階層 + E–vz 耦合 +
  早停判據（z 慢爬被殺）。frequency/phase 模型：stiff z 軸活在振盪空間，v0 設振幅/相位、E 設頻率，
  缺一個就 mismatch → v0_z 與 E 需 joint。
- **scalar wrong-E**（`scalar_uniE5_fitE{4,6}`）：v0_y 正中 strip 預言線（−0.439 / −0.608 vs 預言
  −0.45 / −0.60），但 fitE6 的 vz 飆到 1.28——偏硬時 z 弱軸成為地形外逃生口（3-DOF 就會漏，
  非 field 專利）。

---

## 3. 影像監督：視角即可觀測性（"3DGS 壞梯度" 否定）

工具 `image_fit_ours.py`：pseudo-GaussianModel 建在 MPM 粒子上、單/多固定相機、自洽 roundtrip
（object-only render，比 PhysDreamer 原設定乾淨；distill/真實資料階段要重驗）。

- **scalar v0 可學**：y-front 0.1%、x-front 0.3%（推翻 "±y 純不可辨識"——3-DOF 下透視/平均足夠）。
- **field v0 在病態視角發散**：front-y `imgfield_uniform_y` relL2 1.0→2.9 單調惡化、loss 卻在降——
  pixel loss 的 null space 跑馬，複現 warp 端 "pixel 單視角 ~8 DOF 上限"。
- **可觀測子空間內 image ≈ traj**：diag45 下 ramp_y xy 7.8%（traj 9.2%）、bend xy 23%（traj 27%）。
  失敗全映射到視角幾何，**不在 rasterizer 梯度品質**——兩大迷思之一銷案。
- **multiview 解**（estimator 原生加總同幀多 view）：`imgjoint_initE4` front 單視角失敗
  （warmup 把 v0_y 學成 +0.366 反號 → joint E 跳水 −97%）；**加一個側視**
  `imgjoint_initE4_mv_front_side` → E +1.04% / v0 1.12%，乾淨救回。
  注腳：**wrong-E 的 warmup 階段對視角要求比 correct-E fit 更苛**（深度 sign 判別依賴動力學，
  E 錯時 front 的 ±y sign 失效；放進像平面 side_x −0.24% / diag45 +3.0% 即修）。

---

## 4. v0 field（DOF 階梯 + starve-freeze + TV + 可觀測性）

`V0VoxelField`：voxel grid + trilinear，AnchoredEstimator.set_v0_field 注入 init_velocities，
梯度經 gic 既有 `init_velocities.backward(gradient=...)` 橋回 grid。

- **DOF 階梯全可行**：64（4³）→ 256（4×4×16 各向異性）→ 4096（16³）。劣化是數值級非災難級。
  各向異性 a16 是關鍵（z 加密不攤薄支撐：median node support 123 vs iso-16 的 16；res 計算上免費，
  代價全在 conditioning）。
- **starve-freeze**：support < 1 particle-equiv 的 node 固定（v0→0、E→留 init）+ mask 梯度；純幾何、
  訓練前可算。a16 凍 202/256、iso16 凍 3957/4096——冗餘 DOF 零浪費。
- **mask-aware TV**：uniform GT 下 TV 是**免費午餐**（平場 TV=0，自動填死區，res4 14.9%→1.7%）；
  non-uniform GT 下 TV **與真實梯度作對**，是 tension（bend 平台 ~27% 一部分是 TV-data 平衡點）。
- **可觀測性 = 近 anchor / 弱觀測軸的死區**：corr(err, dist-to-anchor) = −0.58，< 2 cells 的 19%
  粒子背 41% 誤差；但拆軸後 GT 軸（y）誤差小，殘差住在不可觀測子空間 → **大 err ≠ 失敗**。

---

## 5. Joint v0 + E（階梯：scalar → 一場 → 雙場）

最關鍵的 design 結論。GT uniform y @E1e5、init E ±1 decade、nu 凍 GT、vel 4f / phys 8f。

| 階 | 配置 | E 誤差 | v0 | run |
|---|---|---|---|---|
| J2 | scalar v0 + scalar E（hybrid） | +1.1% / +0.2% | ~0.5% | j2_hybrid_initE{4,6} |
| — | alt ×4（block-coord） | **卡死** −68% / +259% | — | j1_alt_initE{4,6} |
| — | joint 冷啟動 | initE6 +0.5% ✓ / **initE4 −98%**（v0 零 init 時 E 在 v0 長出前亂走 1.7 decade，回升途中被早停截殺） | — | j1_joint_initE{4,6} |
| gate | scalar v0 + **E-field** | E-obs 0.004 dex | 0.6% | efj_uniform_ym |
| rung-2 | **v0-field** + scalar E | +0.6% | xy 4.6% | j3_vfield_Escalar |
| rung-3 | **v0-field + E-field（double）** | E-obs 0.006 dex | xy 8.3% | efdouble_uniform_ym |

- **hybrid（warmup v0 → joint）= 通解**：兩側 init 都收（J2 +1.1%/+0.2%）。warmup 40 足夠
  （8-iter 只到 v0 −0.2 → joint 失敗，是暖機不足非機制）。
- **alt 卡死**：strip 谷 + rel_improve 2% 門檻把谷底慢爬判死；加輪數需上百輪，不划算。
- **joint 冷啟動失敗 = 暖機問題**：warmup 先把 v0 帶進盆地即解。alt vs joint vs hybrid 的失敗
  模式各異（互鎖卡死 / 冷啟動誤入 / 無），記在 Evy_J1_overlay 軌跡疊圖。
- **雙場通**：warmup→joint 撐到 v0-field + E-field 同時學。amortize/雙場願景閉環。

---

## 6. E field（應變驅動可觀測性）

`EVoxelField`（log10-E grid，clamp [4,6.15] 保 CFL，uniform init 不可隨機）+ set_E_field 注入
per-particle mu/lam（梯度經 init_mu.backward 回 grid）。

- **E 可觀測性 = 應變驅動，非幾何**。用 `strain_proxy`（k 近鄰相對距離變化，剔除剛體）——
  **位移是錯代理**（懸臂自由端位移大但應變小，anchor 端反之）。
- traj 結果：uniform `efj/efdouble`（E-obs 0.004–0.006 dex）；**ramp** `efdbl_ramp_ym`
  E-obs **0.025 dex（≈6% E）** + v0-ramp xy 10.7%——兩個 ramp 同時學成。
- **circular E（branch-dependent，本輪新）**：電話線兩股，同 z slice 下左右股不同 E（沿弧長漸增）。
  - 表示性先量化驗證（2-means 每 z-slice）：兩股**乾淨可分**（sep-ratio 5–18、中心間隙中位
    1.37 dx、無交叉）。我先前「不可表示」是 median-split artifact，**錯**。
  - **GT = res-16 grid bucket**（不是 per-particle）：bucket 後內插回粒子誤差 0.004 dex、
    中段兩股仍差 0.5 dex → 零表示 floor，recovery 量得乾淨。圖 `circularE_gt_bucketed_res16.png`。
  - fit（res 16³、x 加密分得開兩股）+ uniform-v0（隔離可學性）/ ramp-v0（雙非均勻）跑中。

---

## 7. 跨場景移植（carnations）+ 多觀測

- **carnations SOTA 套餐零調參全收斂**（probe 自動選 +x 激發）：E fit +0.06%、v0 scalar 0.12%、
  v0 field a16 xy 8.3%、hybrid joint E +0.29%。telephone 配方直接通用。
- **M0 多觀測共享 E**（`multiobs_fit.py`，序列化兩 v0、E 梯度累加）：機制正確且穩定，但
  telephone+traj 下單方向已 well-conditioned（單 +x 即 +0.1%），2-obs 無額外增益。
  **amortize 的價值在「單觀測會失敗」的情境**（E field × 多激發覆蓋全場、病態視角互糾）。

---

## 8. 方法論 / 工具沉澱

- **視覺化班底**（每個 field run 必備，別退化成散 png）：3D+triplane overlay（外插段橙標）、
  field/E 雙排 proj（值 / err）、grid-node 切片（support / starved）、profile_1d、panel.gif 總覽。
  panel 統一進 `make_panel.py`（認 scenario 收集預存圖）。
- **ckpt 紀律**：每 N iter 存參數 + optimizer state + grid；可視化升級**永遠離線重建不重跑**；
  失效 run 歸檔不刪；重跑前歸檔同名 log。
- **GPU policy**：擠輕載卡（疊鄰居）、留 idle 給別人；N 卡 > M idle 觸發 (N+0.2)×N；
  serial 常比並行划算。
- **接縫契約**（待辦）：scene 契約 = 6 資料欄位 + canonical orientation（rot 67.6 應是 scene 屬性
  非 run arg）；GT/訓練樣本契約 = {scene, traj (T,N,3) 正規化, 已知/待 fit 參數} → F0 整合退化成
  一次性 adapter。整合痛 = 接縫契約沒寫下來，非架構爛——治接縫別治實作。

---

## 9. 開放問題 / 下一步

- circular E 的**可學性**（≠ 可表示）：單觀測下兩股是否都應變夠、trilinear 在 1.37dx 間距會否滲糊
  （res-16 jobs 跑中）。
- 雙非均勻 + image（目前雙場只在 traj 驗）。
- non-uniform init 的 robustness probe（能否逃出錯誤結構）——尚未跑。
- distill / 真實資料下 object-only render 假設、TV 強度隨 GT 粗糙度調。
