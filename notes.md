






我覺得現在更重要的是兩個:
- 在 traj 上推到 E,v0 joint training
- 把 fixed E 推到 image loss 上。

建議:
1. 再跑兩個 fixed E: ramp y 和 ramp x, 證明對於這種相對平滑的分布也能學
2. 把 fixed 的五個場景: uniform,ramp x/y, bend, mid 跑到 image 上
3. 在 traj 上推 v0, E joint training. 具體來說, 可能要先看 wrong E init 時 v0 能否收斂。如果 v0 不收斂, 後面又凍 v0 學 E
根本浪費時間。如果收斂 => 維持 warmup 策略, E 期待收斂 (v0 已知下); 如果不收斂, 再思考 joint training 策略。


1. 換方向：GT v0 = +x（前視相機下 x 是螢幕水平方向，完全在像平面內）；
2. 換相機：GT 維持 −y，相機搬到側面（(−1.4, 0.5, 0.5) 朝 +x 看，y 變成像平面內方向）。



好, 那到這邊先做一個總結:
  - Spectral loss 解決 beat plateau, 在 R=2,3,5 上得到驗證 (換 R = 換 amplitude)
  - init 太高會因為 Nyquist 走錯方向, 為預期失敗
  - 之所以 Pac nerf (和前面的許多 works) 沒有出這個問題, 是因為當時的 case 中 amplitude 是 E dependent , 因此時域好學; 但現在
  amplitude 被我們的 F0 訂死, 等於 E 只能靠 frequency fit, 剛好是時域弱項。

  總結是否有誤解?

● 沒有誤解,三點都對。只補兩個精確度註腳,讓措辭站得住:

  1. Spectral 解 beat plateau,R=2/3/5(=換 amplitude)驗證 ✓
  完全對。補一句框架:spectral 解的是「beat plateau」(時域飽和,一個 loss-design 問題),它沒解「Nyquist」(一個 sampling-rate
  問題)——這是兩個獨立的失敗模式,你在第 2 點剛好把它們分開了,正確。

  2. init 太高因 Nyquist 走錯方向 = 預期失敗 ✓(但目前是「強推論」,還沒實證釘死)
  我們手上的證據是:(a) 週期算術 logE5.5→2.0 frame = 剛好踩 Nyquist 牆;(b) spectral init5.5 的簽名——loss 有在降(1.8e-1→2e-2)但
  logE 卻往 5.94 跑(離 GT 更遠)= 混疊給了「自信卻指錯方向」的梯度。但「純粹是 Nyquist」這件事要等 5.0/4.75
  跑了才算確認(若那兩個解析得到的 init 能收回
  4.5,就坐實;若收不到,代表上方還有「峰不重疊→無梯度」的第二層問題)。所以這點正確,只是現在是 prediction 等驗證。

  3. PAC-NeRF amplitude E-dependent、我們被 F0 訂死、E 只剩 frequency = 時域弱項 ✓
  對。唯一想把「為什麼 amplitude 會 E-dependent」收成一個更硬的不變量:激發是力控制還是位移控制。PAC-NeRF 用力(重力/接觸)激發 →
  形變量 = 力/勁度 ∝ 1/E → amplitude 帶 E;我們用位移(F0)激發 → amplitude 由 F0 釘死、與 E
  無關。這是根因,你那句是它的結果層描述,等價。

  → 由此來的可證偽預測:drop+collision 把「力」接回來 → 時域應該復活。這就是下一個關鍵 case,同時也是 PAC-NeRF 對帳的實證。

---

當初訂的我再回想一下:
- "field 能否學?" -> 基本上可以, 至少以 4x4x16 可以。
- "img 能否學 field?: -> 補做 scalar + field, 確認 "視角對即可以"。
- "joint training?" -> E 不對 v0 也學不好, pac nerf warmup 是在自由落體下的簡化。必然需要 joint / alternative training.

所以下一個階段:
- A. 確認 field dof 的極限在哪: 測 16x16x16 on uniform + ramp. Why? 1) 確認我們之前 "starved node 不吃記憶體" 的 claim 2)
確認冗餘 dof 的 robustness. 理論上要可以推, 且之前看過有人是拿 per particle 參數訓練, 相較之下 1/16 不應該就崩。
- B. img 這條: 補做剛才的 45 deg, 最後一塊 "只要看得到那 img 和 traj 能力一樣" 的拼圖, 正是否定 "3dgs 讓 grad 壞掉" 的說法。
- C. 探索 joint training. 這是現在最具探索性 (而非 verify 性質) 的一條。有甚麼建議的 strategy? 我感覺應該要退回 scalar v0 + E
的 case。欸但是我忘記之前那個 "mismatch E 下 v0 train 不起來" 的 run 在哪裏了


---


❯ 他在跑的時候我們來聊一下: F0 的訓練到底可以怎麼做?

  參考 v0 那邊的經驗, 我們是先用 scalar against uniform 來驗. 你這邊嘗試用 alpha 做, 但他和 v0 scalar 的簡化不是同一個等級,
  他需要 F0 Gt, 本質上已經不是簡化而是 probe/oracle. 需要想一個真的是合法 (事先不知道 GT) 的簡化。

隨便想到幾個:
- 利用 F 的分解來達成 sharing
- 把 F 拆成 global baseline + local delta.
- SDF 之類的方法

但說實話我根本不確定這些是否適合用在這, 請不要給他們太多 weights, 只是當作潛在靈感。

---

- vel  = 4
- img  = 8
- traj = 14



c to f; early frames