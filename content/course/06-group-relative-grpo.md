# 第 6 章：同一道题采样多次，能不能不用 critic

## 6.1 本章符号说明

| 符号 | English | 本章含义 |
|---|---|---|
| \(x\) | prompt | 一道题或一条指令 |
| \(y_i\) | group response | 对同一 prompt 采样的第 \(i\) 条回答 |
| \(G\) | group size | 每个 prompt 的采样回答数 |
| \(R_i\) | sequence reward | verifier 给第 \(i\) 条完整回答的分数 |
| \(\bar R\) | group mean | 同组回答的平均分 |
| \(s_R\) | group standard deviation | 同组 reward 的标准差 |
| \(\widehat A_i\) | group-relative advantage | 第 \(i\) 条回答相对同组水平的标准化分数 |
| \(r_{i,t}(\theta)\) | importance ratio | current/old 对第 \(i\) 条回答第 \(t\) 个 token 的概率比 |
| \(\Delta_{i,t}\) | reference log-ratio | \(\log\pi_{\text{ref}}-\log\pi_\theta\) |
| \(m_{i,t}\) | completion mask | 第 \(i\) 条回答第 \(t\) 个位置是否是真实生成 token |
| \(\epsilon\) | clip range | 复用 rollout 时允许 ratio 自由变化的局部范围 |
| \(\beta\) | KL coefficient | 相对 reference policy 的惩罚强度 |

## 6.2 本章目标

读完本章，你应该能够：

1. 从“同一道题采样多条回答”得到不依赖 critic 的 baseline；
2. 手算 group mean、标准差与 group-relative advantage；
3. 说明序列级 advantage 广播到 token 不等于获得了逐步监督；
4. 区分只更新一次的 GR-REINFORCE 与复用 rollout 的 GRPO；
5. 从 \(\Delta=\log\pi_{\text{ref}}-\log\pi_\theta\) 理解 sampled KL estimator；
6. 说清 GRPO、PPO、DPO 的数据接口和目标差异；
7. 写出一个完整的 group-relative RLVR 训练循环。

## 6.3 本章主线

第 5 章已经确定：数学题可以用 verifier 给完整回答 \(0/1\) reward。训练时，我们通常不会让一个 prompt 只生成一次，而会采样一组回答：

```text
同一道题 x
 ├─ y1 → 391 → reward 1
 ├─ y2 → 381 → reward 0
 ├─ y3 → 391 → reward 1
 └─ y4 → 无法解析 → reward 0
```

第 3 章用 critic 估计“当前前缀的正常水平”。但这里同一 prompt 已经有 \(G\) 次独立尝试，它们的平均 reward 就能估计：

> 当前模型做这道题，通常能得多少分？

因此本章沿着这条线前进：

\[
\text{同题多次采样}
\rightarrow
\text{组均值 baseline}
\rightarrow
\text{组内相对 advantage}
\rightarrow
\text{单次更新}
\rightarrow
\text{受控复用}.
\]

## 6.4 本章新增概念

| 名词 | 中文直觉 | 本章中解决的问题 |
|---|---|---|
| group sampling | 同一 prompt 采样多条回答 | 同时获得探索与题目难度参照 |
| group baseline | 同组回答平均分 | 不训练 critic 也能构造正常水平 |
| group-relative advantage | 相对同题其他回答的表现 | 去除题目难度尺度 |
| GR-REINFORCE | 组相对 REINFORCE | rollout 只更新一次时的简洁目标 |
| sampled KL estimator | 用已采样 token 估计 reference KL | 不遍历整个词表计算 KL |
| GRPO | 带 ratio 与 clipping 的组相对策略优化 | 在 group advantage 下安全复用 rollout |

## 6.5 从题目难度得到 group baseline

假设 batch 中有两道题：

- 题 A 很简单，8 条回答中 7 条正确；
- 题 B 很难，8 条回答中只有 1 条正确。

若只用绝对 reward，所有正确回答权重都是 \(1\)。但题 B 的一次成功比题 A 的一次成功更稀有；所有题 A 的失败也不应与题 B 的失败用同一参照解释。

对同一 prompt 的 \(G\) 个 reward：

\[
R_1,R_2,\ldots,R_G,
\]

计算：

\[
\bar R
=
\frac1G
\sum_{i=1}^{G}R_i,
\]

\[
s_R
=
\sqrt{
\frac1G
\sum_{i=1}^{G}
(R_i-\bar R)^2
}.
\]

然后定义：

\[
\widehat A_i
=
\frac{R_i-\bar R}
{s_R+\varepsilon}.
\]

\(\bar R\) 是该 prompt 在当前采样温度和策略下的经验正常水平；除以标准差让不同 reward 尺度的组更可比。

## 6.6 完整数值例子

对 `23×17` 采样四条回答，reward 为：

\[
[1,1,0,0].
\]

组均值：

\[
\bar R
=
\frac{1+1+0+0}{4}
=0.5.
\]

组标准差：

\[
s_R
=
\sqrt{
\frac{
(1-0.5)^2+
(1-0.5)^2+
(0-0.5)^2+
(0-0.5)^2
}{4}
}
=0.5.
\]

因此 advantage 为：

\[
\widehat A
=
[1,1,-1,-1].
\]

含义清楚：

- 两条正确回答比同题平均水平好，提高其中采样 token 的概率；
- 两条错误回答比平均水平差，降低其中采样 token 的概率。

再看两种边界：

### 6.6.1 全部答错

\[
R=[0,0,0,0],
\quad
\bar R=0,
\quad
s_R=0.
\]

此时所有 \(\widehat A_i=0\)。这一组没有提供“同题下哪个行为更好”的对比，通常对 policy gradient 没有贡献。它告诉我们探索不足或题目过难，但不能凭空指出改进方向。

### 6.6.2 全部答对

同理，若 \(R=[1,1,1,1]\)，组内也没有排序信息。对 policy 更新贡献为零，但可以用于成功率统计。若训练后大量组都零方差，说明任务可能过易或采样多样性不足。

## 6.7 为什么 group baseline 不需要 critic

第 3 章的 critic 试图学习：

\[
V^\pi(s_0)
=
\mathbb E[R\mid x].
\]

对同一 prompt 直接采样 \(G\) 次，组均值就是这个期望的 Monte Carlo 估计：

\[
\bar R
\approx
V^\pi(x).
\]

因此可以用 \(R_i-\bar R\) 代替 \(R_i-V_\phi(x)\)。代价也同样明确：

- 参照只在 prompt 级别，不能随中间前缀变化；
- \(G\) 小时组均值噪声较大；
- 每个 prompt 必须采多条回答，增加生成成本；
- 没有 critic 的 dense value signal。

它不是“critic 永远无用”，而是用横向的同题多样本，替代纵向的逐前缀 value 预测。

## 6.8 序列级 advantage 怎样作用到 token

verifier 给完整回答 \(R_i\)，所以 \(\widehat A_i\) 也是序列级的。最简单做法是将它广播到该回答的所有有效 completion token：

\[
\mathcal L_i
=
-
\widehat A_i
\frac{
\sum_{t=1}^{T_i}
m_{i,t}
\log\pi_\theta(a_{i,t}\mid s_{i,t})
}{
\sum_{t=1}^{T_i}m_{i,t}
}.
\]

\(m_{i,t}\) 是 completion mask。按有效 token 取平均可以避免长回答仅因 token 多而产生更大梯度。

但必须诚实说明：

> 把一个序列 advantage 复制到每个 token，只是选择了如何分配梯度权重，并没有证明每个 token 同样有功或有错。

在错误回答 `23×17=230+151=381` 中，前面的 `23×17=` 也会收到负 advantage。真正的过程信用分配仍可通过 process reward、搜索、critic 或更细粒度 verifier 改进。

## 6.9 只使用一次 rollout：GR-REINFORCE

若 rollout 由 current policy 生成，并且每批只做一次更新，不需要 old/current importance ratio。目标就是组相对的 REINFORCE：

\[
\mathcal L_{\text{GR-RF}}
=
-
\mathbb E_{i,t}
\left[
\widehat A_i
\log\pi_\theta(a_{i,t}\mid s_{i,t})
\right].
\]

再加 reference KL：

\[
\mathcal L
=
\mathcal L_{\text{GR-RF}}
+
\beta\mathcal L_{\text{KL}}.
\]

这一路线结构简单：

1. current policy 对每个 prompt 采样 \(G\) 条回答；
2. verifier 打分；
3. 组内标准化；
4. 做一次 optimizer update；
5. 丢弃样本，重新采样。

只要数据未被重复使用，先引入 ratio 和 PPO clipping 反而会遮蔽主线。

## 6.10 不遍历词表怎样估计 reference KL

完整 token 分布的 KL 为：

\[
D_{\mathrm{KL}}
\left(
\pi_\theta(\cdot\mid s)
\|
\pi_{\text{ref}}(\cdot\mid s)
\right)
=
\sum_a
\pi_\theta(a\mid s)
\log
\frac{
\pi_\theta(a\mid s)
}{
\pi_{\text{ref}}(a\mid s)
}.
\]

直接计算需要保留整个词表分布。训练中常只保存已采样 token 的 current/ref log-prob。

令：

\[
\Delta
=
\log\pi_{\text{ref}}(a\mid s)
-
\log\pi_\theta(a\mid s).
\]

对 \(a\sim\pi_\theta\)，使用：

\[
k(\Delta)
=
e^\Delta-\Delta-1.
\]

为什么它估计 KL？取 current policy 下的期望：

\[
\begin{aligned}
\mathbb E_{\pi_\theta}
[e^\Delta-\Delta-1]
&=
\mathbb E_{\pi_\theta}
\left[
\frac{\pi_{\text{ref}}(a\mid s)}
{\pi_\theta(a\mid s)}
\right]
-
\mathbb E_{\pi_\theta}[\Delta]
-1\\
&=
1
+
\mathbb E_{\pi_\theta}
\left[
\log
\frac{\pi_\theta(a\mid s)}
{\pi_{\text{ref}}(a\mid s)}
\right]
-1\\
&=
D_{\mathrm{KL}}
\left(
\pi_\theta\|\pi_{\text{ref}}
\right).
\end{aligned}
\]

而且对任意 \(\Delta\)，由 \(e^\Delta\ge1+\Delta\)：

\[
e^\Delta-\Delta-1\ge0.
\]

这比单样本的 \(\log\pi_\theta-\log\pi_{\text{ref}}\) 更适合作为非负诊断量。

注意采样分布：上面的无偏关系要求动作来自 \(\pi_\theta\)。若反复更新 old rollout，样本来自 \(\pi_{\text{old}}\)，严格使用时还要考虑分布校正；工程实现常把它作为局部近似或在 old 与 current 接近时使用。

## 6.11 想重复使用 rollout：自然回到 ratio 与 clipping

若同一组昂贵回答要训练多个 mini-batch/epoch，第 4 章的问题重新出现。于是保存 old log-prob，定义：

\[
r_{i,t}(\theta)
=
\frac{
\pi_\theta(a_{i,t}\mid s_{i,t})
}{
\pi_{\text{old}}(a_{i,t}\mid s_{i,t})
}.
\]

将 group-relative \(\widehat A_i\) 放进 PPO clipped surrogate：

\[
L_{\text{GRPO}}(\theta)
=
\mathbb E_{i,t}
\left[
\min
\left(
r_{i,t}(\theta)\widehat A_i,
\operatorname{clip}
\left(
r_{i,t}(\theta),
1-\epsilon,
1+\epsilon
\right)
\widehat A_i
\right)
-
\beta k(\Delta_{i,t})
\right].
\]

这就是 GRPO 的核心结构：

- **G**：同 prompt 的 group sampling；
- **R**：reward 在组内变成 relative advantage；
- **PPO**：用 ratio 与 clipping 受控复用 old rollout；
- 不训练逐前缀 critic。

GRPO 不是“GAE 换成组均值”这么简单，也不等于 DPO。它仍是在线 rollout、verifier/reward、策略梯度与受控数据复用。

## 6.12 GRPO、PPO、DPO 放在同一坐标系

| 方法 | 数据来自哪里 | 更新权重 | 是否在线 | critic | old policy | reference |
|---|---|---|---:|---:|---:|---:|
| PPO Actor–Critic | current rollout | GAE advantage | 是 | 是 | 是 | 常有 |
| GR-REINFORCE | 同题 current rollout | 组内相对 reward | 是 | 否 | 不需要 | 常有 |
| GRPO | 同题 old rollout | 组内相对 reward + clipped ratio | 是 | 否 | 是 | 常有 |
| DPO | 固定偏好对 | preferred/rejected 相对 margin | 否 | 否 | 否 | 是 |

最关键的分类问题是：

1. 数据是固定偏好对，还是当前模型在线生成？
2. reward 来自人类偏好模型，还是可验证程序？
3. baseline 来自 critic，还是同题组均值？
4. rollout 只用一次，还是要多轮复用？

回答完这四问，算法结构基本就确定了。

## 6.13 完整训练流程

```python
for prompts in dataloader:
    # 1. old policy 对每个 prompt 采样 G 条回答
    rollout = sample_group(
        policy=old_policy,
        prompts=prompts,
        group_size=G,
    )

    # 2. verifier 只根据完整回答给序列 reward
    reward = verifier(rollout.prompt, rollout.response)

    # 3. 在每个 prompt 的组内标准化
    advantage = group_normalize(reward)

    # 4. 保存这批数据的固定量
    rollout.old_logp = old_policy.sampled_token_logp(rollout)
    rollout.ref_logp = reference.sampled_token_logp(rollout)
    rollout.advantage = advantage.detach()

    # 5. 单次更新可用 GR-REINFORCE；多轮复用使用下列 GRPO
    for minibatch in reuse(rollout, update_epochs):
        current_logp = actor.sampled_token_logp(minibatch)
        ratio = exp(current_logp - minibatch.old_logp)

        pg1 = ratio * minibatch.advantage[:, None]
        pg2 = clip(ratio, 1-eps, 1+eps) * minibatch.advantage[:, None]
        policy_objective = minimum(pg1, pg2)

        delta = minibatch.ref_logp - current_logp
        sampled_kl = exp(delta) - delta - 1

        loss = -masked_mean(
            policy_objective - beta * sampled_kl,
            minibatch.completion_mask,
        )
        optimize(loss)

    # 6. current 变成下一批的 old，旧 rollout 丢弃
    old_policy.load_state_dict(actor.state_dict())
```

## 6.14 贯穿案例中的一次更新

对 `23×17` 的四条回答：

| 回答 | reward | group advantage |
|---|---:|---:|
| `230+161=391` | 1 | \(+1\) |
| `20×17+3×17=391` | 1 | \(+1\) |
| `230+151=381` | 0 | \(-1\) |
| 无法解析 | 0 | \(-1\) |

第一次更新：

- 两种正确解法中的有效 token 获得正权重；
- 错误运算和不可解析格式获得负权重；
- reference KL 阻止模型仅为 verifier 而变成奇怪、退化的输出。

若第二个 epoch 中某正确回答的 token ratio 已达 \(1.4\)，而 \(\epsilon=0.2\)，该旧样本不会继续奖励概率比超过 \(1.2\) 的部分。下一批重新采样，才能用 current policy 的新行为获得新证据。

## 6.15 本章容易混淆的结论

| 容易误解成 | 正确理解 |
|---|---|
| 组均值是真实 value | 它只是当前小组对 prompt-level value 的 Monte Carlo 估计 |
| 标准化会创造新 reward 信息 | 它只改变相对尺度；全同 reward 的组仍没有方向 |
| 序列 advantage 广播后就是过程监督 | 它仍是结果级信用分配 |
| GRPO 完全不需要任何基准模型 | 它不需要 critic，但通常仍需要 old 和 reference |
| sampled KL 每个 token 都等于真实 KL | 单样本是估计量，期望才对应分布 KL |
| GRPO 与 DPO 都有 reference，所以相同 | 一个在线优化 verifier reward，一个离线优化偏好对 |

## 6.16 本章自测

1. 为什么同一道题全部答错时，group-relative policy gradient 通常为零？
2. group baseline 与 critic value 各自利用了哪个方向的样本信息？
3. 为什么序列 reward 广播到 token 后仍存在信用分配问题？
4. 何时 GR-REINFORCE 不需要 old policy？何时必须引入 ratio？
5. 推导 \(\mathbb E_{\pi_\theta}[e^\Delta-\Delta-1]=D_{\mathrm{KL}}(\pi_\theta\|\pi_{\text{ref}})\) 的关键一步是什么？
6. 用四个分类问题区分 PPO、GRPO 和 DPO。

## 6.17 本章之后还缺什么

到这里，核心数学链已经闭合。但大模型 RL 训练最常见的失败并不是公式写错，而是：

- logit 与 token 错一位；
- prompt token 混进 policy loss；
- padding 或终止 mask 错；
- `old_logp` 在训练中被意外重算；
- verifier 的解析失败被当成数学错误；
- rollout 更新多轮后仍无限复用。

第 7 章会把前六章翻译成明确的张量契约、数据生命周期和可执行伪代码。

## 6.18 对应论文与资料

- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning](https://arxiv.org/abs/2402.03300)
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [Approximating KL Divergence](http://joschu.net/blog/kl-approx.html)
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948)
