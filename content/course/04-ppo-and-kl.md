# 第 4 章：rollout 很贵，怎样安全地多用几次

## 4.1 本章符号说明

| 符号 | English | 本章含义 |
|---|---|---|
| \(\pi_{\text{old}}\) | behavior / old policy | 生成当前这批 rollout 时冻结的策略快照 |
| \(\pi_\theta\) | current policy | 正在被优化器更新的策略 |
| \(\pi_{\text{ref}}\) | reference policy | 用来约束语言行为不应漂移过远的固定参考模型 |
| \(r_t(\theta)\) | importance ratio | 当前策略与 old 策略对已采样 token 的概率比 |
| \(r_t^{\text{task}}\) | task reward | 环境或 verifier 在第 \(t\) 步给出的任务评分 |
| \(\widehat A_t\) | advantage estimate | 第 3 章得到的 token 更新方向和强度 |
| \(\epsilon\) | clip range | PPO 允许概率比自由变化的局部区间半径 |
| \(\beta\) | KL coefficient | 偏离 reference policy 的惩罚强度 |
| \(D_{\mathrm{KL}}\) | KL divergence | 两个 token 分布之间的平均对数概率差异 |

## 4.2 本章目标

读完本章，你应该能够：

1. 区分 old、current、reference 三个策略的职责；
2. 从“数据由 old 生成、损失却要评价 current”推出 importance ratio；
3. 用正、负 advantage 分别解释概率比过大或过小为什么危险；
4. 从限制单批数据上的更新幅度理解 PPO clipping；
5. 说明 clipping 不是参数距离或 KL 的硬约束；
6. 区分 current–old 的数据复用约束与 current–reference 的行为约束；
7. 写出一轮完整的 PPO rollout—update 生命周期。

## 4.3 本章主线

第 3 章已经构造出更稳定的 advantage。若每采一批 rollout 只更新一次，然后立刻丢弃，REINFORCE 或 Actor–Critic 就能工作。

但大模型 rollout 很贵：它要自回归生成很多 token，还可能调用 verifier、reward model 或工具环境。我们自然希望同一批样本做多个 mini-batch、多个 epoch。

矛盾随即出现：

1. rollout 是参数快照 \(\pi_{\text{old}}\) 生成的；
2. 第一次 optimizer step 后，模型变成 \(\pi_\theta\)；
3. 后续 step 仍在训练旧模型采到的 token；
4. 模型变得越多，这批数据越不像 current policy 自己会生成的数据。

本章只沿着这个矛盾前进：

\[
\text{旧数据}
\rightarrow
\text{概率比校正}
\rightarrow
\text{概率比可能爆炸}
\rightarrow
\text{PPO clipping}
\rightarrow
\text{参考模型 KL}.
\]

## 4.4 本章新增概念

| 名词 | 中文直觉 | 本章中解决的问题 |
|---|---|---|
| on-policy | 用当前策略自己的数据学习 | 分布与优化目标一致，但样本一次性使用 |
| off-policy correction | 对旧策略数据进行分布校正 | 让 old rollout 能估计 current policy 的目标 |
| importance ratio | current 概率除以 old 概率 | 衡量已采样 token 在当前模型下变得多常见 |
| surrogate objective | 可用旧数据估计的替代目标 | 避免每个 optimizer step 都重新 rollout |
| clipping | 截断过激的收益 | 限制单批旧数据继续推动策略的程度 |
| reference KL | 相对初始行为的偏离代价 | 防止模型为 reward 大幅破坏原有语言能力 |

## 4.5 三个策略先分清

同一训练时刻可能同时存在三套 log-prob：

| 角色 | 是否更新 | 何时计算 | 作用 |
|---|---|---|---|
| \(\pi_{\text{old}}\) | 一批 rollout 内冻结 | 生成 token 时 | 记录数据真正来自哪个分布 |
| \(\pi_\theta\) | 持续更新 | 每个训练 step 重新前向 | 被优化的模型 |
| \(\pi_{\text{ref}}\) | 长期冻结 | rollout 后或训练前 | 衡量模型偏离初始/SFT 行为多少 |

刚开始训练一批数据时，current 通常复制自 old，所以两者相同；更新之后才逐渐分开。reference 可能是初始 SFT 模型，也可能是某个确定的基准 checkpoint。它解决的不是数据时效，而是行为漂移。

如果实现中只把三个张量都命名为 `log_probs`，概念一定会混乱。建议明确使用：

```text
old_logp      生成当前样本时保存
current_logp  每次更新重新计算
ref_logp      固定参考模型计算
```

## 4.6 为什么自然会出现概率比

先看只有两个可能答案的简化例子。old policy 生成：

| 回答 | \(\pi_{\text{old}}\) | reward |
|---|---:|---:|
| `391` | 0.25 | 1 |
| `381` | 0.75 | 0 |

更新后 current policy 变为：

| 回答 | \(\pi_\theta\) |
|---|---:|
| `391` | 0.50 |
| `381` | 0.50 |

若仍直接平均 old 数据上的 reward，样本中正确答案只约占四分之一，估计的是 old policy 的表现。要估计 current policy，正确答案样本应被加权：

\[
\frac{\pi_\theta(\text{`391`})}
{\pi_{\text{old}}(\text{`391`})}
=
\frac{0.50}{0.25}
=2.
\]

错误答案样本则乘：

\[
\frac{0.50}{0.75}
=\frac23.
\]

一般地，对任意函数 \(f(a)\)：

\[
\begin{aligned}
\mathbb E_{a\sim\pi_\theta}[f(a)]
&=
\sum_a \pi_\theta(a)f(a)\\
&=
\sum_a \pi_{\text{old}}(a)
\frac{\pi_\theta(a)}{\pi_{\text{old}}(a)}
f(a)\\
&=
\mathbb E_{a\sim\pi_{\text{old}}}
\left[
\frac{\pi_\theta(a)}{\pi_{\text{old}}(a)}
f(a)
\right].
\end{aligned}
\]

对自回归生成的第 \(t\) 个已采样 token，定义：

\[
r_t(\theta)
=
\frac{
\pi_\theta(a_t\mid s_t)
}{
\pi_{\text{old}}(a_t\mid s_t)
}
=
\exp
\left(
\log\pi_\theta(a_t\mid s_t)
-
\log\pi_{\text{old}}(a_t\mid s_t)
\right).
\]

它不是 reward，字母 \(r\) 只是 ratio。为避免与环境 reward 混淆，代码中最好命名 `ratio`。

## 4.7 未裁剪的替代目标

用 old rollout 估计 current policy 的局部策略改进，可以写成：

\[
L^{\text{PG}}(\theta)
=
\mathbb E_t
\left[
r_t(\theta)\widehat A_t
\right].
\]

这里 \(\widehat A_t\) 通常由 rollout 阶段的 reward 和 value 计算并冻结。优化 current policy 时，梯度只穿过 \(r_t(\theta)\) 中的 current log-prob。

概率比可以直接读：

- \(r_t=1\)：current 对该 token 的概率与生成时相同；
- \(r_t=1.2\)：概率变成 old 的 \(1.2\) 倍；
- \(r_t=0.7\)：概率降到 old 的 \(70\%\)。

## 4.8 为什么只做概率比校正还不安全

假设一个 token 的 \(\widehat A_t=2\)，说明应提高概率：

| ratio | \(r_t\widehat A_t\) |
|---:|---:|
| 1.0 | 2.0 |
| 1.2 | 2.4 |
| 3.0 | 6.0 |
| 10.0 | 20.0 |

只要还在重复优化这批数据，模型就会继续从同一个偶然成功样本获利，哪怕它的概率已经被推高十倍。

若 \(\widehat A_t=-2\)，说明应降低概率：

| ratio | \(r_t\widehat A_t\) |
|---:|---:|
| 1.0 | -2.0 |
| 0.8 | -1.6 |
| 0.1 | -0.2 |
| 0.001 | -0.002 |

最大化这个目标会喜欢更接近 \(0\) 的负数，于是可能继续把某个 token 的概率压到几乎为零。

真正的问题不是“ratio 数学上错了”，而是有限批 rollout 的 advantage 有噪声。模型不应无限相信同一批旧证据。

## 4.9 从有限信任得到 PPO clipping

设允许概率比自由变化的区间为：

\[
[1-\epsilon,1+\epsilon].
\]

PPO 使用：

\[
L^{\text{clip}}(\theta)
=
\mathbb E_t
\left[
\min
\left(
r_t(\theta)\widehat A_t,
\operatorname{clip}
\bigl(r_t(\theta),1-\epsilon,1+\epsilon\bigr)
\widehat A_t
\right)
\right].
\]

先不要背 `min`。分别看 advantage 的符号。

### 4.9.1 advantage 为正

若 \(\widehat A_t>0\)，提高采样 token 概率是好事，但 ratio 超过 \(1+\epsilon\) 后不再从该样本获得额外收益。

取 \(\widehat A_t=2,\epsilon=0.2\)：

| ratio | 原目标 | 裁剪目标 | 取较小值 |
|---:|---:|---:|---:|
| 0.8 | 1.6 | 1.6 | 1.6 |
| 1.1 | 2.2 | 2.2 | 2.2 |
| 1.5 | 3.0 | 2.4 | 2.4 |

### 4.9.2 advantage 为负

若 \(\widehat A_t<0\)，降低概率是好事，但 ratio 低于 \(1-\epsilon\) 后也不再获得额外收益。

取 \(\widehat A_t=-2,\epsilon=0.2\)：

| ratio | 原目标 | 裁剪目标 | 取较小值 |
|---:|---:|---:|---:|
| 1.1 | -2.2 | -2.2 | -2.2 |
| 0.9 | -1.8 | -1.8 | -1.8 |
| 0.5 | -1.0 | -1.6 | -1.6 |

由于优化器最大化目标，\(-1.6\) 比 \(-1.0\) 更保守，阻止模型继续从过度降低概率中获利。

统一使用 `min` 正是为了让正、负 advantage 都选择更悲观的收益。

实现中通常最小化负目标：

\[
\mathcal L_{\text{policy}}
=
-
L^{\text{clip}}(\theta).
\]

## 4.10 clipping 限制了什么，又没有限制什么

PPO clipping 限制的是：

> 对这批已经采样的 token，目标函数是否还奖励概率比继续向有利方向移动。

它**不是**：

- 每个参数变化量的硬上界；
- 所有词表 token 概率比都位于区间内的保证；
- current 与 old 的 KL 一定小于某值的保证；
- current 不会偏离初始语言模型的保证。

所以训练仍应监控：

- `approx_kl`：current 与 old 在采样 token 上的近似差异；
- `clip_fraction`：有多少有效 token 触发裁剪；
- ratio 的分位数；
- entropy；
- 梯度范数。

若 `clip_fraction` 很高，常见原因不是“PPO 正在更努力学习”，而是学习率、epoch 数或 advantage 尺度让一批数据被推得太远。

## 4.11 两种 KL 约束不要混在一起

### 4.11.1 current–old：数据是否过期

\[
D_{\mathrm{KL}}
\left(
\pi_{\text{old}}\;\|\;\pi_\theta
\right)
\]

反映 current 相对生成当前 rollout 的 old policy 改了多少。它影响这批数据还能否安全复用，通常用于：

- early stop 当前 PPO epoch；
- 诊断学习率或 update epoch 是否过大；
- 与 ratio、clip fraction 一起监控。

### 4.11.2 current–reference：行为是否漂移

\[
D_{\mathrm{KL}}
\left(
\pi_\theta\;\|\;\pi_{\text{ref}}
\right)
\]

反映训练后的模型偏离固定 SFT/reference 模型多少。它常作为 reward 惩罚：

\[
r_t^{\text{total}}
=
r_t^{\text{task}}
-
\beta
\left(
\log\pi_\theta(a_t\mid s_t)
-
\log\pi_{\text{ref}}(a_t\mid s_t)
\right).
\]

二者对比如下：

| 比较对象 | 时间尺度 | 主要目的 | 一批更新后是否变化基准 |
|---|---|---|---|
| current vs old | 当前 rollout 批次 | 防止旧数据被过度使用 | 下一批会刷新 old |
| current vs reference | 整个训练过程 | 保留原有语言行为、防 reward hacking | reference 通常长期固定 |

old 和 reference 在第一步可能恰好相同，但职责仍完全不同。

## 4.12 一轮完整 PPO 生命周期

现在可以把所有对象按时间顺序放回训练流程：

1. **冻结 old snapshot**
   将当前 actor 作为 \(\pi_{\text{old}}\)。

2. **生成 rollout**
   保存 prompt、completion、`old_logp`、终止标记。

3. **获得反馈**
   verifier 或 reward model 给 reward；reference model 给 `ref_logp`。

4. **计算训练目标**
   critic 产生 value；由 reward、value、终止标记计算 GAE 和 value target。

5. **多轮 mini-batch 更新**
   current actor 重新计算 `current_logp`，形成 ratio 和 clipped objective；critic 回归 value target。

6. **监控批次是否过期**
   检查 approximate KL、clip fraction、ratio 分位数，必要时提前结束 epoch。

7. **丢弃 rollout 并刷新 old**
   不能在 current 已经明显变化后无限复用同一批样本。

伪代码：

```python
old_policy.load_state_dict(actor.state_dict())
rollout = sample(old_policy, prompts)

with no_grad():
    rollout.ref_logp = reference.logp(rollout)
    rollout.reward = scorer(rollout)
    rollout.value = critic(rollout.states)
    rollout.advantage, rollout.value_target = gae(rollout)

for epoch in range(update_epochs):
    for minibatch in iterate(rollout):
        current_logp = actor.logp(minibatch)
        ratio = exp(current_logp - minibatch.old_logp)

        unclipped = ratio * minibatch.advantage
        clipped = clip(ratio, 1-eps, 1+eps) * minibatch.advantage
        policy_loss = -masked_mean(min(unclipped, clipped))

        value_loss = critic_loss(minibatch)
        kl_to_ref = masked_mean(current_logp - minibatch.ref_logp)

        loss = policy_loss + c_v * value_loss + beta * kl_to_ref
        optimize(loss)

        if approx_kl_too_large():
            break
```

## 4.13 回到贯穿案例

对于 `23×17`：

- old policy 采到正确片段 `230+161`，保存其 `old_logp=-1.20`；
- 更新后 current 给同一片段 `current_logp=-0.90`；
- ratio 为 \(\exp(-0.90+1.20)\approx1.35\)；
- 若 advantage 为正且 \(\epsilon=0.2\)，目标只按 \(1.2\) 的收益计算，不再鼓励这条旧样本继续把概率推得更高；
- 同时若 current 比 reference 更偏好机械地重复答案或产生奇怪格式，reference KL 会带来额外代价。

PPO 没有判断 `230+161` 为什么正确。这个信息仍来自 reward 和 advantage。PPO 只解决：**用一批有限、带噪的旧样本更新时，不要走得太猛。**

## 4.14 本章容易混淆的结论

| 容易误解成 | 正确理解 |
|---|---|
| ratio 是 reward | ratio 是 current/old 的采样概率校正 |
| PPO 会把 ratio 硬限制在区间内 | 它只裁剪目标收益；实际 ratio 仍可能越界 |
| old policy 就是 reference policy | old 管数据时效，reference 管长期行为漂移 |
| KL 都是同一回事 | 比较对象不同，训练含义就不同 |
| PPO 可以无限复用 rollout | 数据与 current 差异过大后仍必须丢弃 |
| PPO 自动解决信用分配 | token 权重仍来自 reward、value 和 advantage |

## 4.15 本章自测

1. 为什么第二个 PPO epoch 开始时，rollout 已经不是 current policy 的 on-policy 数据？
2. `current_logp - old_logp = \log 1.5` 表示什么？
3. 当 advantage 为负时，为什么 ratio 过小也要停止奖励？
4. clipping 为什么不是 KL 的硬约束？
5. current–old KL 和 current–reference KL 分别在回答什么问题？
6. 一批 rollout 结束后，哪些量必须保存，哪些量要在每次 optimizer step 重新计算？

## 4.16 本章之后还缺什么

前四章一直假设“某个打分器会给回答 reward”。但真实的大模型训练中，反馈可能来自：

- 人类提供的标准答案；
- 两个回答之间的偏好；
- 学出来的 reward model；
- 能自动核验正确性的程序。

第 5 章会先按反馈数据的形态分类，再由此得到 SFT、reward modeling、RLHF、DPO 与 RLVR。这样算法名称不会先于问题出现。

## 4.17 对应教材与资料

- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [Spinning Up: PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
- [Sutton & Barto, Chapter 5: Off-policy Prediction via Importance Sampling](http://incompleteideas.net/book/RLbook2020.pdf)
- [InstructGPT](https://arxiv.org/abs/2203.02155)
