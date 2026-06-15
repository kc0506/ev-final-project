## Abstract

- Traditional system identification methods treat objects as homogeneous material, and often consider simplified initial condition where object move toward a single direction uniformly. We pose a novel problem: can we recover complex initial condition, on a non-uniform material?
- In addition, system identification with differentiable physics engine often suffer from local minima and vanishing gradient. We propose a novel spectral loss and analyze the landscape of objective functions across several setups.

## Motivation

- Traditional methods has several limitations:
    - Assume object to be rest at the begining => how can we learn an object dynamics that are *released from deformed state* at the begining?
    - Only learns scalar => can we learn field to model more complex setup?
    - Do not always predict ground truth stably. When is the system actually identifable?
    - Use L2 or similar losses. We found that they are not good enough for oscillations.
    - Some methods learn initial velocity and material independently. We jointly train them, and analyze the regime where one of them is dominant.


這裡應該要一張表: past paper x modality supported.

## Method

Main pipeline: 
1. Scene (= multi view with camera position) -> 3dgs particle
2. 3dgs particles are splitted to foreground and background
3. Foreground -> differentiable MPM with input trainables:
    - v0 -> 
    - (hat(u) ->) F0 -> 
    - E
    (use grid to represent field trainables)
    (or maybe use "voxel grid", "MLP" as modules, and have "x_0" as input query)

4. MPM rollout + background + GT video -> loss computation.
    -> frame wise L2 loss
    -> spectral loss (visualization)


### Learning for $F_0$

- \hat u_0 (x) = u(X, 0)
- x_0 = X + u(X, 0) = X + \hat u_0 (x_0)
- F_0 = (\par x_0 / \par X) = (\par X / \par x_0)^{-1} = grad_x_0 u_0(x_0)

We use MLP to parametrize \hat_u_0, and use autograd to compute F_0

<!-- ---------------------------- a: F0 related ---------------------------- -->

## Results: Released from deformed state

- release / drop / asym / ufield
- show per frame comparison
- show gt vs. pred stretch intensity
- show rest state & u prediction

- 選項
    - 保守: F0 已知, 純學 E (+ E field)
    - 激進: E 已知, 學 F0
    - 最激進: E + F0 joint training
    - ultra 激進: E + v0 + F0, 全部 field

- 還缺什麼?
    - image loss + 要 show 的 case
    - scene: telephone, carnation.
        - 這兩個都挺適合做 bend 的。可以各挑兩個 bend 的彎曲錨點


## Results: Inhomogeneous velocity and material distribution

- 如果前面 F0 conditioned 夠激進, 這兩段是不是可以整合一起?

- Bend/mid/ramp x ramp/circular
- per frame traj compare

- show some loss curve (one is enough), and several profile matching (choose pretty ones)
- show learned E heatmap / v0 quiver

- 還缺甚麼?
    - 實驗本身 setup 其實差不多 ok. 剩下全部組合拿到 img 上測。
    - 但其實未必要全部 Cartesian 放 (沒重點)。挑表現好 + 代表性
    - profile 是重點, match 看起來漂亮


## Analysis: Amplitude vs. Frequency

- Choose two setup comparison, and draw their 2d landscape
- One: E is dominant (freq); the other: v/F is dominant, but E is also effective.

- Conclusion: E, (nu) is dominant for frequency, while v/F only affect amplitude.

- 還缺甚麼?
    - 理論上要拿到 img. 但這裡不需要 visual demo => 用 traj 講解 is fine.

## Analysis: Spectral loss

- Compare {time, spectral} x {amplitude, freq} scenes.
- Brief formula showing time L2 loss when freq mismatch is bound and suffer from local minima
- Show the spectrum of different E, and T propto 1/sqrt(E)
- Showing the effects of Nyquist.

- 還缺甚麼?
    - 核心問題: 怎麼推 spectral 到 img 上? 大問題還沒解決!


我想說的是, 之所以我認為 image loss 該推, 是因為我們其他環節都建立在 image loss 上. 因此如果我們能把 freq 這邊的故事 +
成功收斂推到 image 上, 整體邏輯是最通順的。而退一步來說, 如果做不到, 那這部分在方法上是脫節的; 但不代表要捨棄, 而是更往
analysis/probe 那邊偏一點。width 作為 case study 是個合理的 feature, 而使用 3d 這件事也可以說 "我們發現連 oracle
表現都不佳的因素"。



## Augmented learning for non-identifiable scenario.

- show that vy is hard to learn when light axis pointing to +y.
- show that adding another view solve this problem.
