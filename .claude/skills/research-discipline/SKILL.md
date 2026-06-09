---
name: research-discipline
description: Whenever you have to create / inrepret research experiments, or just chat about ideas.
---


### 實作紀律
- 永遠使用 `uv` 或 `mamba` 建環境
- Research code != sloppy code. 每個函數都要 return type hint. 加這個不會造成你 research 效率下降。
- PyTorch tensor 永遠需要註解形狀。Function docstring 也一併註明 input/output 的 shapes

### 用語、專有名詞

- 當你聽到一個你不是很確定意思, 並非行業慣用語的名詞, 永遠向用戶確認意思。舉例: *爆炸* -> 不是指視覺上真的爆炸, 而是 *數值穩定性不好*
- 同理, 對於一個範圍太廣的詞, 而你必須依賴他進行實作, 則請用戶縮小可能的意思。舉例: *coarse-to-fine* -> 什麼模態? 什麼尺度? 哪一種方法?
- 當你自己造詞的時候, 務必向用戶解釋 "我把 XXX 的現象簡稱取名為 YYY"。AI 最常犯的錯就是自己造一堆縮寫講得很開心, 用戶只是看一次覺得煞有介事。

### 結果解讀

- **你用 Read tool 時用戶看不到圖片** 你應該直接告訴我圖片位置，讓我自己去看。
- **不要相信你的圖片能力** 這是最重要的一個條律。每當你說出 "這個結果看起來..." 時，stop。用戶完全不信任任何你對圖片的解讀，無論是正面或負面。任何經由圖片解讀衍生的結論與決策都嚴格被禁止。
  - ❌ : "圖片看起來很合理，XXX 物件出現在了 YYY... 向 user 回報實驗成功"
  - ✅ : "我已經將圖片放在 <path> , 你再確認效果"
- **Always produce gif/mp4** GIF 的可讀性比一張張 frame 好多了。frame png 幾乎永遠是防呆用的 backup, 永遠用可動的影片格式作為主要提供用戶視覺檢查的媒介。
- **Loss curve 是最重要的** AI 看實驗結果最常犯的錯就是看各種 metrics 扯半天, 就是不看 loss。如果 loss 根本沒收斂, 你所有的分析都是空談。你可以對 loss 做初步解讀, 但 **把 loss curve 圖片給 user 看** 是必要操作。人類一眼就能看出有沒有收斂。


### GPU 規則

隨時使用 `gpu-policy` `ws-status` 和 `nvidia-smi` 確認現況。規則: gpu quota >= 12: 最多同時用 2 gpu. gpu quota<12 hr: 最多同時用 1 gpu. gpu quota<4h: 禁止使用 gpu, 暫時終止任務。
  - 注意: gpu quota 的算法是 per user. 我會跑其他 processes -> 你不能依據你的記憶判斷 "現在有哪些gpu在使用, 所以同時用 n 張
  gpu" , 而是永遠在跑之前先確認。
  - 不該 kill 任何 "非你創建, 且已經在跑的 processes".