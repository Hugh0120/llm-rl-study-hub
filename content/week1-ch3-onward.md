# 第三章：从“整份答案得几分”到“这一步比预期好多少”

第二章已经得到一个能训练的规则：

\[
\mathcal L_{\text{PG}}
=
-\mathbb E_t\left[G_t\log\pi_\theta(a_t\mid s_t)\right].
\]

它的逻辑没有错：回报高，就提高已采样 token 的概率；回报低，就降低它的概率。真正的问题是，\(G_t\) 同时混入了两类信息：

1. 这道题本来有多难；
2. 当前这个 token 选得是否比通常更好。

策略更新真正需要的是第二类信息。本章只解决这一件事。路线是：

> 先构造“正常水平”作为参照 → 再学习这个参照 → 再用相邻前缀之间的预测变化分配信用 → 最后才给已经理解的做法加上课本术语。

## 3.1 为什么绝对回报不是理想的更新权重

假设同一批数据里有两道题：

| prompt | 正确率 | 某次回答 reward |
|---|---:|---:|
| \(x_{\text{easy}}\) | 90% | 1 |
| \(x_{\text{hard}}\) | 5% | 1 |

两次回答都得 1 分，但含义不同：

- 简单题答对接近正常发挥；
- 难题答对远高于正常发挥。

如果都用 \(G=1\) 更新，策略看不见这种差别。因此我们先减去一个只由当前状态决定的参照 \(b(s_t)\)：

\[
\widehat A_t=G_t-b(s_t).
\]

先不要急着记 `advantage` 这个名称。把它读成一句中文即可：

> 当前动作之后得到的回报，减去站在当前前缀上原本预计能得到的回报。

于是：

- \(\widehat A_t>0\)：这一步之后的结果好于预期，应提高该 token 的概率；
- \(\widehat A_t<0\)：结果差于预期，应降低该 token 的概率；
- \(\widehat A_t\approx0\)：结果符合预期，更新应很小。

这也解释了为什么 0/1 reward 不能直接读成“错误 token 一定被惩罚”。如果完全不用参照，错误回答的 \(G=0\) 只会让梯度近似为零；有了正的预期分数后，\(0-b(s_t)<0\)，错误回答才会形成明确的负向信号。

## 3.2 减去参照为什么不会把正确梯度改错

我们要求 \(b(s_t)\) 可以依赖状态，但不能偷看当前采样动作。固定一个状态 \(s\)，参照项对梯度的期望为：

\[
\begin{aligned}
\mathbb E_{a\sim\pi_\theta(\cdot\mid s)}
\left[b(s)\nabla_\theta\log\pi_\theta(a\mid s)\right]
&=
b(s)\sum_a \pi_\theta(a\mid s)
\nabla_\theta\log\pi_\theta(a\mid s)\\
&=
b(s)\sum_a\nabla_\theta\pi_\theta(a\mid s)\\
&=
b(s)\nabla_\theta\sum_a\pi_\theta(a\mid s)\\
&=
b(s)\nabla_\theta 1=0.
\end{aligned}
\]

因此：

\[
\mathbb E[(G_t-b(s_t))\nabla\log\pi]
=
\mathbb E[G_t\nabla\log\pi].
\]

参照没有改变期望梯度，只减少了样本之间无关的起伏。这就是它能“降方差而不系统性改方向”的原因。

> 边界：如果参照直接依赖当前动作 \(a_t\)，上面的最后一步通常不成立。那已不再是普通 baseline，需要重新证明。

## 3.3 最自然的参照：站在当前前缀上，平均能得多少分

理想参照不是随便一个常数，而是当前策略从状态 \(s\) 继续生成时的期望回报：

\[
V^\pi(s)
=
\mathbb E_{\tau\sim\pi}\left[G_t\mid s_t=s\right].
\]

\(V^\pi(s)\) 叫作状态价值。对 LLM：

- \(s_t\)：prompt 加已经生成的 token；
- \(V^\pi(s_t)\)：从这个前缀继续采样，最终平均能得到多少回报。

如果还固定当前动作，就得到动作价值：

\[
Q^\pi(s,a)
=
\mathbb E_{\tau\sim\pi}\left[G_t\mid s_t=s,a_t=a\right].
\]

两者之差才正式叫优势：

\[
A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s).
\]

现在这个术语不再突兀：**Advantage 就是某动作相对当前状态正常水平的超额表现。**

### 数值例子：同样答对，更新强度不同

假设 value 预测：

\[
V(s_{\text{easy}})=0.9,\qquad V(s_{\text{hard}})=0.05.
\]

两次回答都得 \(G=1\)，则：

\[
\widehat A_{\text{easy}}=1-0.9=0.1,
\qquad
\widehat A_{\text{hard}}=1-0.05=0.95.
\]

难题中的成功轨迹得到更强的强化，正是因为它更“出乎预期”。

## 3.4 value 不知道答案：把它变成一个监督学习问题

真实的 \(V^\pi\) 是未知期望。我们用一个带参数的预测器 \(V_\phi(s)\) 逼近它。

最直接的训练标签，是 rollout 已经发生后算出的实际回报 \(G_t\)：

\[
\mathcal L_V(\phi)
=
\mathbb E_t\left[
\left(V_\phi(s_t)-G_t\right)^2
\right].
\]

这就是普通回归：

- 输入：当前前缀 \(s_t\)；
- 标签：从这里往后实际拿到的 \(G_t\)；
- 输出：预测的期望回报。

到这里才需要两个角色名：

- **Actor**：产生动作的策略 \(\pi_\theta(a\mid s)\)；
- **Critic**：评估状态的预测器 \(V_\phi(s)\)。

所谓 **Actor–Critic** 没有多出一套神秘算法，它只是：

1. critic 学习“正常水平”；
2. actor 根据“实际表现减正常水平”更新。

二者的 loss 不同，参数也可以不同：

\[
\mathcal L_{\text{actor}}
=
-\mathbb E_t[\widehat A_t\log\pi_\theta(a_t\mid s_t)],
\]

\[
\mathcal L_{\text{critic}}
=
\mathbb E_t[(V_\phi(s_t)-\widehat V_t^{\text{target}})^2].
\]

## 3.5 不等整条轨迹结束：先推出一步预测误差

用完整 \(G_t\) 训练 value 是无偏的，但长序列里方差大，而且必须等终局。我们希望只看一步：

> 当前状态的价值，应该等于这一步奖励，加上下一个状态的价值。

这就是 Bellman 一致性：

\[
V^\pi(s_t)
=
\mathbb E\left[r_t+\gamma V^\pi(s_{t+1})\mid s_t\right].
\]

把期望换成这次真实转移，并把真实 value 换成预测器，得到一步目标：

\[
\widehat V_t^{(1)}
=
r_t+\gamma V_\phi(s_{t+1}).
\]

“目标减当前预测”是：

\[
\delta_t
=
r_t+\gamma V_\phi(s_{t+1})-V_\phi(s_t).
\]

先读含义，再记名字：

> 走完这一步以后，新的信息让我们把最终回报预期上调了多少或下调了多少。

这个量叫 **temporal-difference error，TD 误差**。`temporal difference` 指相邻时间步价值预测之间的差，不是另一种 reward。

### 数值例子：一个 token 如何获得局部信号

设 \(\gamma=1\)，生成某 token 前：

\[
V(s_t)=0.45.
\]

生成后还未结束，\(r_t=0\)，但新前缀更像正确解，value 变为：

\[
V(s_{t+1})=0.60.
\]

于是：

\[
\delta_t=0+0.60-0.45=0.15.
\]

这个 token 得到正信号，不是因为它立刻拿到了 reward，而是因为它让“最终成功概率”的预测上升了。

若最后输出错误，终止状态的后续价值记为 0，且 \(r_T=0\)。假设结束前 \(V(s_T)=0.55\)，则：

\[
\delta_T=0+0-0.55=-0.55.
\]

终局失败会把先前过于乐观的预测拉回去。

## 3.6 一步很稳但有偏，整段很准但很抖

一步目标依赖 \(V_\phi(s_{t+1})\)。预测器尚不准确时，它会把自己的误差带回训练，这叫 **bootstrapping（自举）**。

另一个极端是一直看到终局：

\[
\widehat V_t^{(\text{MC})}
=
r_t+\gamma r_{t+1}+\cdots+\gamma^{T-t}r_T.
\]

它不依赖未来 value 预测，偏差小，但不同 rollout 的结果波动大。

中间方案是 \(n\)-step target：

\[
\widehat V_t^{(n)}
=
\sum_{l=0}^{n-1}\gamma^l r_{t+l}
+\gamma^nV_\phi(s_{t+n}).
\]

| 目标 | 看多远 | 主要优点 | 主要代价 |
|---|---:|---|---|
| 1-step | 1 步 | 方差较小、更新及时 | 更依赖 critic，偏差可能大 |
| \(n\)-step | \(n\) 步 | 可调折中 | 多一个尺度选择 |
| Monte Carlo | 到终局 | 不从未来 value 自举 | 长轨迹方差大 |

这一步先建立“不同视野长度”的概念，下一节才组合它们。

## 3.7 GAE：把不同视野的 TD 信息做指数加权

我们已经有每一步的预测修正 \(\delta_t\)。可以把未来若干步的修正累加：

\[
\widehat A_t^{(1)}=\delta_t,
\]

\[
\widehat A_t^{(2)}=\delta_t+\gamma\delta_{t+1},
\]

\[
\widehat A_t^{(3)}=\delta_t+\gamma\delta_{t+1}
+\gamma^2\delta_{t+2}.
\]

希望近处权重大、远处逐渐衰减，就得到：

\[
\widehat A_t^{\text{GAE}(\gamma,\lambda)}
=
\sum_{l=0}^{T-t}
(\gamma\lambda)^l\delta_{t+l}.
\]

实现时通常反向递推：

```python
gae = 0.0
for t in reversed(range(T)):
    nonterminal = 1.0 - done[t]
    delta = reward[t] + gamma * value[t + 1] * nonterminal - value[t]
    gae = delta + gamma * lam * nonterminal * gae
    advantage[t] = gae

value_target = advantage + value[:-1]
```

\(\lambda\) 控制“相信一步 critic”还是“更多相信真实后续”：

- \(\lambda=0\)：退化为一步 TD，通常方差低、偏差高；
- \(\lambda\to1\)：接近长视野回报，通常偏差低、方差高。

`done` 不是实现细节。终止后没有下一个真实状态，必须切断自举，否则会把下一条样本的 value 串进本条轨迹。

## 3.8 只有终局 reward 时，GAE 到底做了什么

数学推理常见：

\[
r_t=0\quad(t<T),\qquad r_T=R.
\]

这不意味着所有 token 天生拥有精确的过程监督。GAE 的局部差异来自 critic：

- 某个前缀让成功概率预测上升，附近 \(\delta_t\) 为正；
- 某个前缀暴露错误，让预测下降，附近 \(\delta_t\) 为负；
- 终局 reward 再修正整条预测链。

所以必须说清边界：

1. GAE 能把**学到的前缀价值变化**传播成 token 级权重；
2. 它不会凭空知道哪一步逻辑是对的；
3. critic 不准时，细粒度 credit assignment 也不准；
4. 如果不训练 critic，而只把同一序列 reward 广播给所有 completion token，那仍是序列级监督。

第二周会介绍另一条路线：对同一 prompt 采样一组回答，用组内均值代替 learned value baseline。那是 GRPO/GR-REINFORCE 的出发点，不要提前与本章混为一谈。

### 第三章检查点

读完应能不看公式回答：

1. baseline 为什么能产生负向更新，却不改变期望梯度？
2. value 是什么量的条件期望？
3. critic 的训练标签从哪里来？
4. TD error 为什么可以读成“预期修正”？
5. GAE 中 \(\lambda\) 在折中什么？
6. 终局 reward 为什么不等于真实过程监督？

# 第四章：同一批 rollout 为什么不能随便多训

第三章改善了每个动作的权重，却仍有一个数据问题。

第二章和第三章的推导都默认：

\[
a_t\sim\pi_\theta(\cdot\mid s_t).
\]

但工程中，rollout 很贵。我们通常先冻结一份策略采样，再对这批数据做多个 minibatch、多个 epoch。第一次更新后，当前策略已经变了，数据却仍来自旧策略。问题就变成：

> 怎样有限度地复用旧数据，同时不让一次更新把策略推得太远？

这才是 PPO 要回答的问题。

## 4.1 三份模型先分清

LLM PPO 常同时出现三份概率，不分清就一定会把 KL 和 ratio 写混：

| 名称 | 记号 | 是否更新 | 用途 |
|---|---|---:|---|
| old policy | \(\pi_{\text{old}}\) | 一轮 rollout 内冻结 | 产生数据，提供旧 log-prob |
| current policy | \(\pi_\theta\) | 是 | 本轮正在优化 |
| reference policy | \(\pi_{\text{ref}}\) | 通常冻结 | 约束模型别偏离 SFT 起点太远 |

`old` 是短期快照，每轮会刷新；`reference` 是长期行为锚点。二者偶尔参数相同，也不代表概念相同。

## 4.2 从换分布估计推出概率比

想计算当前策略下某个量 \(f(a)\) 的期望，却只有旧策略样本：

\[
\begin{aligned}
\mathbb E_{a\sim\pi_\theta}[f(a)]
&=
\sum_a\pi_\theta(a\mid s)f(a)\\
&=
\sum_a\pi_{\text{old}}(a\mid s)
\frac{\pi_\theta(a\mid s)}
{\pi_{\text{old}}(a\mid s)}
f(a).
\end{aligned}
\]

定义概率比：

\[
r_t(\theta)
=
\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\text{old}}(a_t\mid s_t)}
=
\exp\left(
\log\pi_\theta(a_t\mid s_t)
-\log\pi_{\text{old}}(a_t\mid s_t)
\right).
\]

于是旧样本上的策略目标可写成：

\[
\mathcal L_{\text{surrogate}}(\theta)
=
\mathbb E_t[r_t(\theta)\widehat A_t].
\]

直觉：

- \(r_t=1\)：当前策略对该 token 的概率没变；
- \(r_t=1.2\)：概率变为旧策略的 1.2 倍；
- \(r_t=0.7\)：概率变为旧策略的 0.7 倍。

> 限制：这个比值修正了给定旧状态 \(s_t\) 时的动作概率，没有完整修正更新后策略访问到的状态分布。策略变化太大时，旧前缀本身也会失去代表性。因此还需要限制步长。

## 4.3 为什么直接优化概率比会失控

若 \(\widehat A_t>0\)，最大化 \(r_t\widehat A_t\) 会持续增大 \(r_t\)；若 \(\widehat A_t<0\)，会持续减小 \(r_t\)。在同一批数据上训练很多次，模型可能极端放大少量偶然成功样本。

PPO 的核心是：超出一个局部区间后，不再从这批旧样本获得额外“便宜收益”。

\[
\mathcal L_{\text{clip}}(\theta)
=
\mathbb E_t\left[
\min\left(
r_t\widehat A_t,\;
\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)\widehat A_t
\right)
\right].
\]

训练代码通常最小化负号：

\[
\mathcal L_{\text{actor}}=-\mathcal L_{\text{clip}}.
\]

## 4.4 分正负两种情况读 clip

设 \(\epsilon=0.2\)。

### 好动作：\(\widehat A=2\)

若 \(r=1.35\)：

\[
r\widehat A=2.7,\qquad
\operatorname{clip}(r,0.8,1.2)\widehat A=2.4.
\]

取较小值 2.4。概率已经提高很多后，继续提高不会改善 clipped objective。

### 坏动作：\(\widehat A=-2\)

若 \(r=0.70\)：

\[
r\widehat A=-1.4,\qquad
\operatorname{clip}(r,0.8,1.2)\widehat A=-1.6.
\]

取较小值 \(-1.6\)。概率已经降低很多后，继续降低也不会得到额外好处。

| advantage | 策略想做什么 | 哪一侧被截断 |
|---|---|---|
| \(\widehat A>0\) | 提高该动作概率 | \(r>1+\epsilon\) |
| \(\widehat A<0\) | 降低该动作概率 | \(r<1-\epsilon\) |

PPO 不是把所有 ratio 都硬裁成区间，而是用 `min` 构造悲观目标。区间另一侧仍会保留惩罚性梯度。

## 4.5 clip 不等于真正的距离保证

clip 只作用于样本中的动作概率比，不保证整个词表分布的 KL 一定小。因此至少监控：

\[
D_{\mathrm{KL}}
\left(
\pi_{\text{old}}(\cdot\mid s)
\;\|\;
\pi_\theta(\cdot\mid s)
\right).
\]

它回答：“本轮 current 离 rollout 快照走了多远？”

另一个常见 KL 是：

\[
D_{\mathrm{KL}}
\left(
\pi_\theta(\cdot\mid s)
\;\|\;
\pi_{\text{ref}}(\cdot\mid s)
\right).
\]

它回答：“训练后的模型离长期 reference 走了多远？”

| KL | 角色 | 典型用途 |
|---|---|---|
| old ↔ current | 优化稳定性 | early stop、调小 epoch/学习率 |
| current ↔ reference | 行为锚定 | 奖励塑形或额外正则 |

不要只写“监控 KL”而不说明是哪两份策略。

<details>
<summary>可选：TRPO 与 natural gradient 在这条故事中的位置</summary>

TRPO 直接提出一个带 KL 约束的优化问题，在局部二阶近似下得到 natural-gradient 风格的方向。PPO 用一阶优化和 clipping/KL penalty 给出更容易实现的近似。对本周主线，只需记住：TRPO 更显式地约束分布距离，PPO 更工程化，但 clip 不是数学上的硬信赖域。

</details>

### 第四章检查点

1. 为什么第一次更新后 rollout 就变成旧数据？
2. old policy 与 reference policy 的生命周期有什么不同？
3. ratio 从哪个换分布等式得到？
4. 为什么正 advantage 和负 advantage 的截断方向相反？
5. PPO clip 为什么不能代替 KL 监控？

# 第五章：把 value、GAE、PPO 接成一轮可执行训练

现在所有组件都有了来由。本章只做一件事：明确数据在什么时候由谁产生，以及哪些量必须冻结。

## 5.1 一轮训练的因果顺序

### 步骤一：冻结 rollout 快照

\[
\pi_{\text{old}}\leftarrow\pi_\theta.
\]

记录每个采样 completion token 的：

- `token_id`
- `old_logp`
- `value`
- `completion_mask`

### 步骤二：生成并打分

对 prompt \(x_i\)：

\[
y_i\sim\pi_{\text{old}}(\cdot\mid x_i),
\qquad
R_i=\text{reward}(x_i,y_i).
\]

reward 可以来自人工、reward model 或 verifier。它负责评价结果，不负责预测未来。

### 步骤三：构造逐步 reward

终局任务最简单的版本：

\[
r_{i,t}=
\begin{cases}
0,&t<T_i,\\
R_i,&t=T_i.
\end{cases}
\]

若使用 reference KL 约束，可选两条实现路径之一：

**路径 A：KL 进入 shaped reward**

\[
r^{\text{shaped}}_{i,t}
=
r^{\text{task}}_{i,t}
-\beta\,k_{i,t}.
\]

随后用 shaped reward 计算 GAE。

**路径 B：KL 作为单独 loss**

\[
\mathcal L
=
\mathcal L_{\text{actor}}
+c_v\mathcal L_{\text{value}}
+\beta\mathcal L_{\text{KL}}.
\]

两条路径的尺度和梯度语义不同。除非你明确推导过，不要同时加两次 KL。

### 步骤四：冻结 advantage 与 old log-prob

用 rollout 阶段保存的 value 和 reward 计算：

\[
\widehat A_t=\operatorname{GAE}(r_t,V_{\text{old}}(s_t)),
\qquad
\widehat R_t=\widehat A_t+V_{\text{old}}(s_t).
\]

进入 PPO epoch 后，`advantage`、`value_target`、`old_logp` 都作为常量使用，不能随着每个 minibatch 重新反向传播。

### 步骤五：有限复用 rollout

```python
for epoch in range(K):
    for mb in minibatches(rollout):
        new_logp = policy.logprob(mb.tokens)
        ratio = exp(new_logp - mb.old_logp)

        unclipped = ratio * mb.advantage
        clipped = clamp(ratio, 1-eps, 1+eps) * mb.advantage
        actor_loss = -masked_mean(min(unclipped, clipped), mb.mask)

        new_value = critic(mb.states)
        value_loss = masked_mean(
            (new_value - mb.value_target) ** 2,
            mb.mask,
        )

        loss = actor_loss + value_coef * value_loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

### 步骤六：丢弃旧数据，重新采样

完成 \(K\) 个 epoch 后，这批 rollout 的复用到此为止。下一轮必须：

\[
\pi_{\text{old}}\leftarrow\pi_\theta
\]

并生成新数据。PPO 是**受限制的短期复用**，不是把 rollout 变成永久离线数据集。

## 5.2 五个模型头分别回答什么

| 组件 | 输入 | 输出 | 训练/冻结 | 回答的问题 |
|---|---|---|---|---|
| current policy | 前缀 | 下个 token 分布 | 训练 | 接下来生成什么？ |
| old policy | 前缀 | 旧 token 概率 | 一轮冻结 | 数据由谁生成？ |
| reference policy | 前缀 | 参考 token 概率 | 长期冻结 | 偏离起点多远？ |
| value/critic | 前缀 | 期望未来回报 | 训练 | 从这里通常能得多少分？ |
| reward/verifier | 完整输入输出 | 标量或规则结果 | 通常固定 | 这份结果有多好？ |

reward model 和 value model 都输出标量，但条件、目标和生命周期不同，不能互换。

## 5.3 贯穿例子：乘法题的一轮 PPO

prompt：`23 × 17 = ?`

old policy 生成两条：

1. `391`，verifier 给 \(R=1\)；
2. `381`，verifier 给 \(R=0\)。

critic 认为生成首位数字前成功率约为 0.35：

- 正确轨迹最终会产生正的 value 修正；
- 错误轨迹终局会产生负的 value 修正；
- GAE 把这些修正按距离传播回各 token；
- PPO 用 old/new ratio 限制同一批样本的复用强度；
- reference KL 防止模型为了这类题而整体语言分布漂移过远。

这条链里没有哪个组件能独自完成全部工作：

\[
\text{verifier 定义结果}
\rightarrow
\text{critic 估计预期}
\rightarrow
\text{GAE 构造相对权重}
\rightarrow
\text{PPO 限制更新}
\rightarrow
\text{新 policy 产生下一轮数据}.
\]

## 5.4 训练日志应该对应哪一段机制

| 现象 | 先看 | 机制解释 |
|---|---|---|
| reward 不动，entropy 很快塌缩 | 采样多样性、学习率 | 还没探索到成功样本就变确定 |
| clip fraction 长期很高 | epoch、学习率、\(\epsilon\) | 旧数据被推得过远 |
| old-current KL 暴涨 | 更新步数、batch | rollout 很快失效 |
| current-reference KL 暴涨 | \(\beta\)、reward 漏洞 | 策略远离行为锚点 |
| value loss 下降但 reward 不升 | value target、策略梯度 | critic 会拟合数据不代表 actor 变好 |
| explained variance 很差 | mask、终止、自举 | critic 没学会回报结构 |
| 长回答占优 | token sum/mean、长度奖励 | loss 聚合方式引入长度偏差 |

# 第六章：把整周压成一张依赖图

## 6.1 每个概念是为了解决哪个前一个问题

| 已有方法 | 暴露的问题 | 下一概念 |
|---|---|---|
| 轨迹 reward | 不可直接对离散采样求导 | log-derivative trick |
| REINFORCE | 绝对回报噪声大 | baseline |
| baseline | 理想参照未知 | value/critic |
| Monte Carlo value | 长轨迹方差大、反馈晚 | Bellman 自举与 TD |
| 1-step TD | critic 偏差重 | n-step 与 GAE |
| on-policy actor–critic | rollout 贵，只训一次浪费 | importance ratio |
| 直接复用旧数据 | ratio 失控、状态分布漂移 | PPO clip / KL |
| PPO 任务优化 | 可能远离 SFT 行为 | reference KL |

如果一个术语不能落回这张表中的“问题”，说明它还没有真正理解。

## 6.2 六个必须会手算的量

给定：

\[
\log\pi_\theta=-1.2,\quad
\log\pi_{\text{old}}=-1.4,\quad
\widehat A=-0.5,\quad
\epsilon=0.2,
\]

应能算出：

1. \(r=\exp(-1.2+1.4)\)；
2. unclipped objective \(r\widehat A\)；
3. clipped ratio；
4. clipped objective；
5. 两者的 `min`；
6. 最小化代码中的 actor loss 符号。

再给：

\[
r_t=0,\quad \gamma=1,\quad
V(s_t)=0.3,\quad V(s_{t+1})=0.5,
\]

应能算出 \(\delta_t=0.2\)，并解释它为什么不是环境 reward。

## 6.3 本周完成标准

- [ ] 能从期望回报独立推到 REINFORCE；
- [ ] 能证明 state-only baseline 不改变期望梯度；
- [ ] 能用中文区分 return、value、Q、advantage、TD error；
- [ ] 能从一步预测误差推出 GAE 递推；
- [ ] 能解释 old/current/reference 三份策略；
- [ ] 能分 advantage 正负读懂 PPO clip；
- [ ] 能画出 rollout → reward → GAE → PPO → resample；
- [ ] 能说明终局 reward 下 token credit 的能力边界；
- [ ] 能指出 KL 是哪两份模型之间的 KL；
- [ ] 能从日志定位是采样、critic、PPO 还是 reward 出问题。

# 附录：官方课件与对应视频

正文已经自包含。下面只用于复习原始讲授和核对公式。

| 本文位置 | 官方课件 | 对应视频 |
|---|---|---|
| 第一章：MDP、轨迹、回报 | [CS285 L4 · RL Basics](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-4.pdf) | [CS285 Fall 2023 · Lecture 4](https://www.youtube.com/playlist?list=PL_iWQOsE6TfVYGEGiAOMaOzzv41Jfm_Ps&index=4) |
| 第二章：REINFORCE | [CS285 L5 · Policy Gradients](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-5.pdf) | [CS285 Fall 2023 · Lecture 5](https://www.youtube.com/playlist?list=PL_iWQOsE6TfVYGEGiAOMaOzzv41Jfm_Ps&index=5) |
| 第三章：value、TD、GAE | [CS285 L6 · Actor–Critic](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-6.pdf) | [CS285 Fall 2023 · Lecture 6](https://www.youtube.com/playlist?list=PL_iWQOsE6TfVYGEGiAOMaOzzv41Jfm_Ps&index=6) |
| 第四章：换分布估计 | [CS285 L9 · Advanced Policy Gradients I](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-9.pdf) | [CS285 Fall 2023 · Lecture 9](https://www.youtube.com/playlist?list=PL_iWQOsE6TfVYGEGiAOMaOzzv41Jfm_Ps&index=9) |
| 第四章：约束、PPO、TRPO | [CS285 L10 · Advanced Policy Gradients II](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-10.pdf) | [课程页中的最新录像入口](https://rail.eecs.berkeley.edu/deeprlcourse/) |

下一周不再重复 PPO 推导，而是回答一个新问题：**reward 从哪里来，以及偏好数据、自动 verifier 和组内比较分别适合什么场景。**
