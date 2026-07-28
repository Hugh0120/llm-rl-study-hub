# 第 3 章：一条回答只有一个分数，怎样判断每一步做得好不好

## 3.1 本章符号说明

| 符号 | English | 本章含义 |
|---|---|---|
| \(G_t\) | return from step \(t\) | 从第 \(t\) 步之后实际拿到的累计回报 |
| \(r_t\) | immediate reward | 第 \(t\) 步动作完成后立刻得到的外部评分 |
| \(V^\pi(s_t)\) | true state value | 按当前策略从前缀 \(s_t\) 继续生成，未来回报的期望 |
| \(V_\phi(s_t)\) | learned value | 由参数为 \(\phi\) 的 value model 对 \(V^\pi(s_t)\) 的近似 |
| \(b(s_t)\) | baseline | 只依赖当前状态、用于减小梯度方差的参照值 |
| \(\widehat A_t\) | estimated advantage | 本次实际结果相对当前前缀正常水平好多少 |
| \(\delta_t\) | TD residual | 看完下一步后，对价值预测的一次修正量 |
| \(\gamma\) | discount factor | 未来 reward 的折扣系数；有限长度推理常取 \(1\) |
| \(\lambda\) | GAE parameter | 在“更信 critic”与“更信完整结果”之间连续调节 |
| \(d_t\) | terminal flag | 第 \(t\) 步后回答是否真正终止 |

## 3.2 本章目标

读完本章，你应该能够：

1. 说明为什么所有 token 共用最终分数虽然正确，却会产生很大的训练噪声；
2. 从“同一前缀继续生成很多次的平均得分”理解 \(V^\pi(s)\)；
3. 说明 value 不是人工填写的标签，而是由 rollout 回报监督学出来的；
4. 从“实际结果减正常水平”自然得到 advantage；
5. 从相邻两次价值预测的修正自然得到 TD residual；
6. 理解 GAE 为什么是多步 TD 修正的加权和，而不是凭空出现的公式；
7. 正确处理完成、截断和 padding，避免把下一条样本串进来。

## 3.3 本章主线

第 2 章已经解决了“最终分数怎样变成模型梯度”：

\[
\mathcal L_{\text{REINFORCE}}
=
-\mathbb E_t
\left[
G_t\log\pi_\theta(a_t\mid s_t)
\right].
\]

这条式子没有错。问题在于：如果数学题只在回答结束时给 \(0/1\) 分，那么一条正确回答里的每个 token 都拿到 \(G_t=1\)，一条错误回答里的每个 token 都拿到 \(G_t=0\)。

考虑两条回答：

```text
A: 23×17 = 23×(10+7) = 230+161 = 391。 \boxed{391}
B: 23×17 = 230+151 = 381。 \boxed{381}
```

对 A 而言，`23×(10+7)` 和 `230+161` 确实帮助了正确结果；但句号、格式 token 和一些可替换措辞也一起拿到同样的正权重。对 B 而言，真正的错误是 `151`，而前面的分解思路并不差；最终的 \(0\) 却无法告诉模型错误发生在哪里。

本章不假设我们突然拥有逐 token 奖励。我们只增加一个可学习的问题：

> 看到当前前缀以后，如果继续按当前模型生成，通常还能拿几分？

这个问题的答案就是 value。先有这个问题，后面才会依次出现 baseline、advantage、TD、GAE 和 Actor–Critic。

## 3.4 本章新增概念

| 名词 | 中文直觉 | 本章中解决的问题 |
|---|---|---|
| value | 当前前缀的未来得分预期 | 区分简单前缀和危险前缀的“正常水平” |
| value model / critic | 预测 value 的模型 | 无法为每个前缀真的续写无数次时，近似这个期望 |
| baseline | 梯度里的参照线 | 减去与动作无关的正常水平，降低方差而不改变期望梯度 |
| advantage | 超出预期的部分 | 决定本次已采样动作应被鼓励还是抑制 |
| TD residual | 一步之后的预测修正 | 用下一状态的 value 提前获得学习信号 |
| GAE | 多尺度 advantage 估计 | 在完整回报的高方差与一步预测的偏差之间折中 |
| Actor–Critic | 策略与价值协作的训练结构 | actor 负责生成，critic 负责提供更稳定的更新权重 |

这些词不是六套互不相干的算法。它们属于一条因果链：

\[
\text{完整回报噪声大}
\rightarrow
\text{需要正常水平}
\rightarrow
\text{value}
\rightarrow
\text{advantage}
\rightarrow
\text{一步修正 TD}
\rightarrow
\text{多步折中 GAE}.
\]

## 3.5 value 到底是什么

仍以当前策略 \(\pi\) 回答 `23×17` 为例。状态 \(s_t\) 是 prompt 加已经生成的前缀。假设我们能从同一个前缀独立续写很多次，并用 verifier 给最终答案打 \(0/1\) 分：

| 当前前缀 | 续写 100 次中答对次数 | 平均最终分数 |
|---|---:|---:|
| `题目：23×17。` | 42 | 0.42 |
| `23×17=23×(10+7)=` | 71 | 0.71 |
| `23×17=230+161=` | 96 | 0.96 |
| `23×17=230+151=` | 3 | 0.03 |

当 reward 只有 \(0/1\) 时，这个平均值也可以读成“从该前缀继续答对的概率”。在一般 reward 下，它仍然是未来回报的期望：

\[
V^\pi(s_t)
=
\mathbb E_{\tau_{t:}\sim\pi}
\left[
G_t\mid s_t
\right].
\]

请注意三个边界：

1. \(V^\pi(s_t)\) 评价的是**前缀所处的局面**，不是某个候选 token；
2. 它依赖策略 \(\pi\)：模型变强后，同一前缀的成功率可能提高；
3. 表中的数字只是帮助理解的抽样估计，实际训练不会为每个前缀专门生成 100 条续写。

### 3.5.1 value 在大模型中怎样计算

工程上通常在语言模型最后一层 hidden state 上接一个标量头：

\[
V_\phi(s_t)
=
w^\top h_t+c.
\]

`h_t` 表示读完当前前缀后的隐藏表示。这个标量头和主模型可以：

- 共享 Transformer 主干，只单独增加 value head；
- 使用独立的 value model；
- 在小规模实验中使用同一模型的两个输出头。

输出头一开始并不知道“0.71”意味着什么。它的监督信号来自已经完成的 rollout。

## 3.6 critic 的训练标签从哪里来

假设一条实际 rollout 在第 \(t\) 步之后最终得到了 \(G_t\)。对当前前缀来说，\(G_t\) 是未来随机回报的一次真实样本。因此可以训练：

\[
\mathcal L_V(\phi)
=
\frac12
\mathbb E_t
\left[
\left(
V_\phi(s_t)-\widehat R_t
\right)^2
\right].
\]

\(\widehat R_t\) 是 value target。最容易理解的 target 就是实际完整回报：

\[
\widehat R_t=G_t.
\]

单条样本会很嘈杂。例如同一个尚可的前缀偶尔因采样失误得到 \(0\)，不能说明它的真实 value 就是 \(0\)。但大量 rollout 上的平方误差最优解正是条件期望：

\[
V_\phi(s)
\approx
\mathbb E[G_t\mid s_t=s]
=
V^\pi(s).
\]

所以，value 不是预先给好的神秘数字，也不是“模型自己相信多少”。它是用历史 rollout 的实际结果做回归，学出来的平均未来回报。

## 3.7 从正常水平得到 advantage

现在比较两个前缀：

- 简单题前缀的正常成功率是 \(0.9\)，这次答对得到 \(1\)；
- 难题前缀的正常成功率是 \(0.1\)，这次也答对得到 \(1\)。

如果都使用原始回报 \(1\)，两次更新强度相同。但第二次结果比预期好得多，包含的学习信息更强。于是用实际结果减去正常水平：

\[
\widehat A_t
=
G_t-V_\phi(s_t).
\]

\(\widehat A_t\) 称为 advantage 估计：

- \(\widehat A_t>0\)：本次结果比当前前缀通常能做到的更好，提高所采样 token 的概率；
- \(\widehat A_t<0\)：本次结果比预期更差，降低其概率；
- \(\widehat A_t\approx0\)：结果符合预期，更新很小。

策略损失变成：

\[
\mathcal L_{\text{actor}}
=
-\mathbb E_t
\left[
\widehat A_t
\log\pi_\theta(a_t\mid s_t)
\right].
\]

### 3.7.1 为什么减 baseline 不会把目标改坏

只要 baseline \(b(s_t)\) 在看到状态后不再依赖“本次采样了哪个动作”，它对期望策略梯度的贡献为零：

\[
\begin{aligned}
&\mathbb E_{a_t\sim\pi_\theta}
\left[
b(s_t)\nabla_\theta
\log\pi_\theta(a_t\mid s_t)
\right]\\
&=
b(s_t)
\sum_{a_t}
\pi_\theta(a_t\mid s_t)
\nabla_\theta\log\pi_\theta(a_t\mid s_t)\\
&=
b(s_t)
\sum_{a_t}
\nabla_\theta\pi_\theta(a_t\mid s_t)\\
&=
b(s_t)\nabla_\theta 1
=0.
\end{aligned}
\]

因此 baseline 不改变“平均而言要优化什么”，只去掉所有动作共享的那部分波动。学习出来的 \(V_\phi(s_t)\) 是常用 baseline，因为它正好预测当前状态的正常回报。

## 3.8 为什么还需要比完整回报更早的信号

\(G_t-V(s_t)\) 已经比原始回报更合理，但仍必须等回答结束才能计算。如果回答很长，前面每个 token 的监督都由遥远的最终结果决定。

生成一个新片段后，我们其实立刻获得了两条信息：

1. 本步是否收到了 reward \(r_t\)；
2. 新前缀 \(s_{t+1}\) 的局面看起来比旧前缀更好还是更坏。

旧预测是 \(V(s_t)\)。看完这一步后，对同一件事的新预测是：

\[
r_t+\gamma V(s_{t+1}).
\]

两者之差定义为一次价值预测修正：

\[
\delta_t
=
r_t+\gamma(1-d_t)V_\phi(s_{t+1})-V_\phi(s_t).
\]

这就是 temporal-difference residual，简称 **TD residual**。名字可以在理解之后再记：它只是“相隔一步的两个预测之差”。

其中 \((1-d_t)\) 很重要：

- 若本步后回答仍在继续，保留下一个前缀的 value；
- 若本步后真正终止，未来已不存在，必须令下一 value 为 \(0\)。

### 3.8.1 把数值代进去

假设一条正确回答只有三步，\(\gamma=1\)，中途 reward 都为 \(0\)，终止步 reward 为 \(1\)：

| \(t\) | 当前前缀 | \(V(s_t)\) | 本步后 \(V(s_{t+1})\) | \(r_t\) | \(\delta_t\) |
|---:|---|---:|---:|---:|---:|
| 0 | 只有 prompt | 0.30 | 0.50 | 0 | \(0+0.50-0.30=0.20\) |
| 1 | 已写出正确分解 | 0.50 | 0.80 | 0 | \(0+0.80-0.50=0.30\) |
| 2 | 已得到 391，随后结束 | 0.80 | 0 | 1 | \(1+0-0.80=0.20\) |

这里的 TD 不是突然引入的一种新 reward。三项只是同一个最终惊喜 \(1-0.30=0.70\) 被沿途的预测修正分开了：

\[
\delta_0+\delta_1+\delta_2
=
0.20+0.30+0.20
=
0.70
=
G_0-V(s_0).
\]

当 \(\gamma=1\) 且 value 边界正确时，中间 value 会在求和中前后抵消。这说明 TD 修正与最终 advantage 属于同一条账，不是凭空多出来的概念。

## 3.9 一步 TD 为什么还不够

直接令 \(\widehat A_t=\delta_t\) 有一个优点：只看下一步，信号及时、方差通常较低。但它也有明显代价：严重依赖 critic 的预测。

例如模型写下正确的 `230+161`，但尚未训练好的 critic 把新前缀从 \(0.5\) 错估成 \(0.45\)。即使这一步很好，\(\delta_t\) 也可能为负。

另一端是完整回报：

\[
\widehat A_t
=
G_t-V(s_t).
\]

它最终相信 verifier，偏差较小；但一条 rollout 的偶然性会一路传回所有 token，方差较高。

我们需要的不是第三个互不相关的方法，而是把这两个端点连续连接起来。

## 3.10 从多步预测修正得到 GAE

从第 \(t\) 步开始，把后续 TD 修正按距离加权：

\[
\widehat A_t^{\text{GAE}(\gamma,\lambda)}
=
\delta_t
+\gamma\lambda\delta_{t+1}
+(\gamma\lambda)^2\delta_{t+2}
+\cdots.
\]

这就是 generalized advantage estimation，简称 **GAE**。它可以直接读成：

> 当前动作的好坏，等于现在看到的预测修正，加上一部分更远处的预测修正。

\(\lambda\) 控制“看多远”：

- \(\lambda=0\)：只用 \(\delta_t\)，最信 critic 的一步判断；
- \(\lambda\rightarrow1\)：纳入几乎全部后续修正，接近 \(G_t-V(s_t)\)；
- 中间值：远处证据按指数衰减，常见设置如 \(0.95\)。

对上一节的三步例子，若 \(\gamma=1,\lambda=0.5\)：

\[
\begin{aligned}
\widehat A_2&=0.20,\\
\widehat A_1&=0.30+0.5\times0.20=0.40,\\
\widehat A_0&=0.20+0.5\times0.40=0.40.
\end{aligned}
\]

若 \(\lambda=1\)，则：

\[
\widehat A_0
=
0.20+0.30+0.20
=
0.70
=
G_0-V(s_0).
\]

这正好验证了两个端点的关系。

## 3.11 value target 怎样与 advantage 配套

得到 \(\widehat A_t\) 后，常用的 value target 是：

\[
\widehat R_t
=
\widehat A_t+V_\phi(s_t).
\]

直觉是：旧预测加上这次应该修正的量，得到新的监督目标。训练时通常把右侧从计算图中 detach，避免 value target 自己追着自己求梯度。

此时两个损失分工明确：

\[
\mathcal L_{\text{actor}}
=
-\mathbb E_t
\left[
\widehat A_t
\log\pi_\theta(a_t\mid s_t)
\right],
\]

\[
\mathcal L_{\text{critic}}
=
\frac12\mathbb E_t
\left[
\left(
V_\phi(s_t)-\widehat R_t
\right)^2
\right].
\]

到这里才有必要给整个结构命名：

- 生成 token 的策略 \(\pi_\theta\) 是 **actor**；
- 预测前缀正常水平的 \(V_\phi\) 是 **critic**；
- actor 使用 critic 提供的 advantage 更新，critic 使用 rollout 结果校准自己。

**Actor–Critic 不是本章开头必须接受的前置概念，而是我们已经推导出的两种角色的简称。**

## 3.12 终止、截断与 padding 必须分开

代码里常见：

```python
delta = reward[t] + gamma * next_value * nonterminal - value[t]
```

这里的 `nonterminal` 更清楚的名字是 `can_bootstrap`：当前样本后面是否存在同一条轨迹的真实下一状态，可以继续使用 \(V(s_{t+1})\)。

三种边界不能混为一谈：

| 情况 | 含义 | 下一 value |
|---|---|---:|
| terminated | 生成 `<eos>`、提交答案或环境真正结束 | \(0\) |
| truncated | 因最大长度或时间预算被外部截断 | 取决于任务定义；若保留未完局面的估值，可 bootstrap |
| padding | 为 batch 对齐补出的空位置 | 不参与任何 loss |

如果数据管线把下一条样本的第一个 value 当作本条终止样本的 `next_value`，GAE 会把两条回答串起来。正确做法是同时维护：

- `loss_mask`：该位置是不是有效 completion token；
- `terminated`：本步后环境是否真正结束；
- 必要时 `truncated`：是否只是被预算切断。

## 3.13 反向递推实现

GAE 的公式看似包含很多未来项，实际只需从后往前扫描：

```python
gae = 0.0

for t in reversed(range(response_length)):
    valid = completion_mask[t]
    can_bootstrap = 1.0 - terminated[t]

    delta = (
        reward[t]
        + gamma * can_bootstrap * value[t + 1]
        - value[t]
    )

    gae = (
        delta
        + gamma * lam * can_bootstrap * gae
    )

    advantage[t] = gae * valid

value_target = advantage + value[:-1]
```

这段代码成立的前提是：

1. `value[t]` 与生成第 \(t\) 个 token 之前的前缀对齐；
2. 终止后的 `value[t+1]` 不会被使用；
3. padding 位置的 actor loss、critic loss 和统计量都被 mask；
4. `advantage` 与 `value_target` 在更新 actor/critic 前已停止梯度。

## 3.14 从大模型视角重新串一次

对 prompt `计算 23×17`：

1. actor 从当前前缀采样下一个 token；
2. verifier 通常只在完整答案结束时给最终 \(0/1\)；
3. critic 在每个前缀上预测“继续生成的平均成功率”；
4. 新 token 产生的新前缀会改变这个成功率，由此得到一步预测修正 \(\delta_t\)；
5. GAE 把当前和未来若干次修正合成 \(\widehat A_t\)；
6. \(\widehat A_t>0\) 时，提高本次 token 的 log-prob；小于 \(0\) 时降低；
7. rollout 的实际结果又反过来训练 critic。

这条循环没有假装知道每个 token 的真实功劳。它只是借助 value 把“整条回答的结果”转换成比统一最终分数更稳定的相对信号。

## 3.15 本章容易混淆的结论

| 容易误解成 | 正确理解 |
|---|---|
| value 是当前答案正确的确定概率 | 它是按当前策略继续采样时的期望回报估计 |
| advantage 是新的外部 reward | 它是实际回报或预测修正相对 baseline 的训练权重 |
| TD 是逐 token reward | 它是相邻价值预测之间的一次修正 |
| GAE 能精确识别哪个 token 犯错 | 它只是多尺度的 advantage 估计，仍可能有信用分配误差 |
| Actor–Critic 是突然换了一套算法 | 它是策略模型和 value 模型协作关系的名称 |
| `done` 与 padding 相同 | 前者表示轨迹边界，后者只是张量对齐 |

## 3.16 本章自测

1. 为什么同样得到最终 \(1\) 分，低 value 前缀上的成功通常具有更大的正 advantage？
2. value model 的训练标签从哪里来？为什么单条 rollout 的 \(0\) 或 \(1\) 不等于真实 value？
3. 用一句话解释 \(\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)\)。
4. 为什么 \(\lambda=0\) 更依赖 critic，\(\lambda\rightarrow1\) 更依赖完整 rollout？
5. 若终止后仍乘入下一条样本的 value，会造成什么错误？
6. 在本章推导顺序中，Actor–Critic 为什么应该最后命名而不是最先介绍？

## 3.17 本章之后还缺什么

现在我们已经有了更合理的 token 更新权重。但还没有回答一个工程问题：

> rollout 是旧参数生成的；优化几步以后，当前模型已经变化。还能继续把这些昂贵样本当作当前模型的数据使用吗？

第 4 章会先从这一个数据时效问题推出 importance ratio，再解释 PPO 的 clipping，以及 old、current、reference 三个模型为什么不能混为一谈。

## 3.18 对应教材与资料

- [Sutton & Barto, Chapter 6: Temporal-Difference Learning](http://incompleteideas.net/book/RLbook2020.pdf)
- [Sutton & Barto, Chapter 13: Policy Gradient Methods](http://incompleteideas.net/book/RLbook2020.pdf)
- [Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)
- [Berkeley CS285: Actor-Critic Algorithms](https://rail.eecs.berkeley.edu/deeprlcourse/)
